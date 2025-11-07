/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

module layernorm #(
	int unsigned  N,
	int unsigned  SIMD
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,

	// (Parallel) Input Stream
	input	logic [SIMD-1:0][31:0]  xdat,
	input	logic  xvld,
	output	logic  xrdy,

	// (Parallel) Output Stream
	output	logic [SIMD-1:0][31:0]  ydat,
	output	logic  yvld,
	input	logic  yrdy
);

	localparam int unsigned  NN = N / SIMD;
	initial begin
		if(N%SIMD != 0) begin
			$error("%m: SIMD(%0d) must divide N(%0d).", SIMD, N);
			$finish;
		end
		if(NN <= 12) begin
			$error("%m: N/SIMD must be larger than 12 for rsqrt throughput.");
			$finish;
		end
	end

	// Until paths of reduction trees are balanced out
	initial begin
		if(2**$clog2(SIMD) != SIMD) begin
			$error("%m: SIMD(%0d) must currently be a power of two.", SIMD);
			$finish;
		end
	end

	typedef logic [31:0]  fp32;
	typedef fp32 [SIMD-1:0] vfp32;
	typedef struct {
		fp32   dat;
		logic  vld;
	} edge_t;
	typedef struct {
		vfp32  dat;
		logic  vld;
		logic  rdy;
	} vedge_t;

	//=======================================================================
	// Build Normalization Diamonds

	// Connectivity: #0 -> Mean Shift -> #1 -> Variance Scaling -> #2
	uwire vedge_t  vedge[3];
	assign	vedge[0].dat = xdat;
	assign	vedge[0].vld = xvld;
	assign	xrdy = vedge[0].rdy;
	assign	ydat = vedge[2].dat;
	assign	yvld = vedge[2].vld;
	assign	vedge[2].rdy = yrdy;

	for(genvar  step = 0; step < 2; step++) begin : genNormDiamonds

		localparam int unsigned  STATISTICS_LATENCY =
			// SIMD adder tree + accumulation + decouple
			$clog2(SIMD) * 2   +     4        +    3 +
			// Variance: *1/N + rsqrt
			step   *    (  3  + 14  );
		localparam int unsigned  VALUE_QUEUE_LEN = NN + STATISTICS_LATENCY;
		localparam int unsigned  STATS_QUEUE_LEN = 2 + (VALUE_QUEUE_LEN-1)/NN;

		//-------------------------------------------------------------------
		// Value bypass Queue
		uwire vedge_t  bypass;
		queue #(.DATA_WIDTH(SIMD*32), .ELASTICITY(VALUE_QUEUE_LEN)) bypass_queue (
			.clk, .rst,
			.idat(vedge[step].dat), .ivld(vedge[step].vld), .irdy(vedge[step].rdy),
			.odat(bypass     .dat), .ovld(bypass     .vld), .ordy(bypass     .rdy)
		);
		// Input pacing solely by bypass queue
		uwire  xtxn = vedge[step].vld && vedge[step].rdy;

		//-------------------------------------------------------------------
		// Free-running Statistics Queue
		uwire vfp32  xdat = vedge[step].dat;
		uwire edge_t  norm;
		if(step == 0) begin : genMean
			// Reduce parallel Inputs into one partial Sum
			uwire edge_t  part_sum;
			if(1) begin : blkInputReduce

				uwire edge_t  tree[2*SIMD-1];
				for(genvar  i = 0; i < SIMD; i++) begin : genLeaves
					assign	tree[SIMD-1+i] = '{ vld: xtxn, dat: xdat[i] };
				end : genLeaves
				for(genvar  i = 0; i < SIMD-1; i++) begin : genNodes
					binopf #(.OP("ADD")) node (
						.clk, .rst,
						.r(tree[i]    .dat), .rvld(tree[i]    .vld),
						.a(tree[2*i+1].dat), .avld(tree[2*i+1].vld),
						.b(tree[2*i+2].dat), .bload(1'b1)
					);
				end : genNodes
				assign	part_sum = tree[0];

			end : blkInputReduce

			// Accumulation of parital Sums
			if(1) begin : blkAccu

				// Identify last Input Transaction
				uwire  alst;
				if(NN == 1)  assign  alst = 1;
				else begin
					logic signed [$clog2(NN-1):0]  Cnt = NN-2; // NN-2, ..., 1, 0, -1
					always_ff @(posedge clk) begin
						if(rst)  Cnt <= NN-2;
						else     Cnt <= Cnt + (!part_sum.vld? 0 : !alst? -1 : NN-1);
					end
					assign	alst = Cnt[$left(Cnt)];
				end

				// Accumulation
				accuf #(.SCALE(1.0/N)) accu (
					.clk, .rst,
					.a(part_sum.dat), .avld(part_sum.vld), .alst,
					.s(norm.dat), .svld(norm.vld)
				);

			end : blkAccu
		end : genMean
		else begin : genVar

			// SIMD parallel partial Accumulation of Squares
			uwire edge_t  part_sq[SIMD];
			if(1) begin : blkSumq

				// Identify last Input Transaction
				uwire  alst;
				if(NN == 1)  assign  alst = 1;
				else begin
					logic signed [$clog2(NN-1):0]  Cnt = NN-2; // NN-2, ..., 1, 0, -1
					always_ff @(posedge clk) begin
						if(rst)  Cnt <= NN-2;
						else     Cnt <= Cnt + (!xtxn? 0 : !alst? -1 : NN-1);
					end
					assign	alst = Cnt[$left(Cnt)];
				end

				for(genvar  i = 0; i < SIMD; i++) begin : genSumq
					accuf #(.SCALE(0.0 /*SQR*/)) sumq (
						.clk, .rst,
						.a(xdat[i]), .avld(xtxn), .alst,
						.s(part_sq[i].dat), .svld(part_sq[i].vld)
					);
				end : genSumq

			end : blkSumq

			// Reduction to a single total Sum scaled by 1/N
			uwire edge_t  vari;
			if(1) begin : blkSqReduce
				edge_t  tree[2*SIMD-1];
				assign	tree[SIMD-1+:SIMD] = part_sq;
				for(genvar  i = 0; i < SIMD-1; i++) begin : genNodes
					binopf #(.OP("ADD")) node (
						.clk, .rst,
						.r(tree[i]    .dat), .rvld(tree[i]    .vld),
						.a(tree[2*i+1].dat), .avld(tree[2*i+1].vld),
						.b(tree[2*i+2].dat), .bload(1'b1)
					);
				end : genNodes
				binopf #(.OP("MUL")) node (
					.clk, .rst,
					.a(tree[0].dat), .avld(tree[0].vld),
					.b($shortrealtobits(1.0/N)), .bload(1'b1),
					.r(vari.dat), .rvld(vari.vld)
				);
			end : blkSqReduce

			// Inverse Square Root
			uwire  vrdy;
			rsqrtf vari_rsqurt (
				.clk, .rst,
				.x(vari.dat), .xvld(vari .vld), .xrdy(vrdy),
				.r(norm.dat), .rvld(norm.vld)
			);
			always_ff @(posedge clk) begin
				assert(rst || !vari.vld || vrdy) else begin
					$error("%m Overrunning rsqrt computation.");
					$stop;
				end
			end
		end : genVar

		//-------------------------------------------------------------------
		// Apply Normalization
		if(1) begin : blkApply

			// Statistics Queue catching all possible Computations in Flight
			uwire edge_t  norm0;
			uwire  norm0_rdy;
			if(1) begin : blkMeanCatcher
				uwire  norm_rdy;
				queue #(.DATA_WIDTH(32), .ELASTICITY(STATS_QUEUE_LEN)) catcher (
					.clk, .rst,
					.idat(norm .dat), .ivld(norm .vld), .irdy(norm_rdy),
					.odat(norm0.dat), .ovld(norm0.vld), .ordy(norm0_rdy)
				);
				always_ff @(posedge clk) begin
					assert(rst || !norm.vld || norm_rdy) else begin
						$error("%m: Overrunning statistics queue.");
						$stop;
					end
				end
			end : blkMeanCatcher

			// Free-Running Normalization Operator bracketed by credit-based Flow Control
			localparam int unsigned  CREDIT = 7;
			logic signed [$clog2(CREDIT):0]  Credit = CREDIT-1; // CREDIT-1, ..., 1, 0, -1
			uwire  have_cap = !Credit[$left(Credit)];
			uwire  issue;
			uwire  settle;
			always @(posedge clk) begin
				if(rst)  Credit <= 6;
				else     Credit <= Credit + (issue == settle? 0 : settle? 1 : -1);
			end

			logic signed [$clog2(NN-1):0]  Cnt = 0;	// [-NN,] -NN+1, ..., -1, 0
			assign	norm0_rdy = !Cnt[$left(Cnt)];
			assign	issue = have_cap && (norm0.vld || Cnt[$left(Cnt)]);
			uwire  bload = norm0.vld && norm0_rdy;
			always @(posedge clk) begin
				if(rst)  Cnt <= 0;
				else     Cnt <= Cnt + (bload? -NN : 0) + issue;
			end
			always_ff @(posedge clk) begin
				assert(rst || bypass.vld || !issue) else begin
					$error("%m: Drained bypass.");
					$stop;
				end
			end
			assign	bypass.rdy = issue;

			uwire vfp32  rdat;
			uwire  rvld;
			for(genvar  i = 0; i < SIMD; i++) begin : genOps
				uwire  rvld0;
				binopf #(.OP(step? "MUL" : "SUB")) op (
					.clk, .rst,
					.a(bypass.dat[i]), .avld(issue),
					.b(norm0.dat), .bload,
					.r(rdat[i]), .rvld(rvld0)
				);
				if(i == 0)  assign  rvld = rvld0;
			end : genOps

			// Output Queue
			uwire  rrdy;
			queue #(.DATA_WIDTH(SIMD * 32), .ELASTICITY(CREDIT)) decouple (
				.clk, .rst,
				.idat(rdat), .ivld(rvld), .irdy(rrdy),
				.odat(vedge[step+1].dat), .ovld(vedge[step+1].vld), .ordy(vedge[step+1].rdy)
			);
			always_ff @(posedge clk) begin
				assert(rst || !rvld || rrdy) else begin
					$error("%m: Overruning normalization output.");
					$stop;
				end
			end
			assign	settle = vedge[step+1].vld && vedge[step+1].rdy;

		end : blkApply

	end : genNormDiamonds

endmodule : layernorm
