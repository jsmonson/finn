/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @brief	Skid buffer with optional feed stages to ease long-distance routing.
 * @todo
 *	Offer knob for increasing buffer elasticity at the cost of allowable
 *	number of feed stages.
 ***************************************************************************/

module skid #(
	int unsigned  DATA_WIDTH,
	int unsigned  FEED_STAGES = 0
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [DATA_WIDTH-1:0]  idat,
	input	logic  ivld,
	output	logic  irdy,

	output	logic [DATA_WIDTH-1:0]  odat,
	output	logic  ovld,
	input	logic  ordy
);

	typedef logic [DATA_WIDTH-1:0]  dat_t;

	uwire  aload;
	uwire dat_t  adat;
	uwire [3:0]  aptr;
	uwire  bvld;
	uwire  bload;
	if(FEED_STAGES == 0) begin : genNoFeedStages

		// Elasticity Control Logic
		logic [1:0]  AVld = '0;
		logic  ARdy = 1;	// = !AVld[1]
		assign	irdy = ARdy;
		assign	bvld = |AVld;

		always_ff @(posedge clk) begin
			if(rst) begin
				AVld <= '0;
				ARdy <= 1;
			end
			else begin
				automatic logic  ardy = !AVld || bload;
				AVld <= '{ !ardy, AVld[1]? AVld[0] : ivld };
				ARdy <= ardy;
			end
		end
		assign	aload = irdy;
		assign	adat = idat;
		assign	aptr = { 3'b000, AVld[1] };

	end : genNoFeedStages
	else begin : genFeedStages

		//- Allow up to 7 plain-forward FEED_STAGES
		initial begin
			if(FEED_STAGES > 7) begin
				$error("%m: Requested %0d FEED_STAGES exceeds support for up to 7.", FEED_STAGES);
				$finish;
			end
		end

		// Dumb input stages to ease long-distance routing
		uwire  ardy;
		if(1) begin : blkInputFeed
			dat_t  IDat[FEED_STAGES] = '{ default: 'x };
			logic  IVld[FEED_STAGES] = '{ default: 0 };
			logic  IRdy[FEED_STAGES] = '{ default: 1 };
			always_ff @(posedge clk) begin
				if(rst) begin
					IDat <= '{ default: 'x };
					IVld <= '{ default: 0 };
					IRdy <= '{ default: 1 };
				end
				else begin
					for(int unsigned  i = 0; i < FEED_STAGES-1; i++) begin
						IDat[i] <= IDat[i+1];
						IVld[i] <= IVld[i+1];
						IRdy[i] <= IRdy[i+1];
					end
					IDat[FEED_STAGES-1] <= idat;
					IVld[FEED_STAGES-1] <= ivld && irdy;
					IRdy[FEED_STAGES-1] <= ardy;
				end
			end
			assign	aload = IVld[0];
			assign	adat = IDat[0];
			assign	irdy = IRdy[0];
		end : blkInputFeed

		// Elasticity Control Logic
		logic signed [$clog2(2*FEED_STAGES+2):0]  APtr = '1;
		assign	ardy = APtr < 1;
		assign	bvld = !APtr[$left(APtr)];

		always_ff @(posedge clk) begin
			if(rst)  APtr <= '1;
			else     APtr <= APtr + $signed((aload == (bload && bvld))? 0 : aload? 1 : -1);
		end
		assign	aptr = $unsigned(APtr[$left(APtr)-1:0]);

	end : genFeedStages

	//-----------------------------------------------------------------------
	// Buffer Memory: SRL:2+2*FEED_STAGES + Reg (no reset)

	// Elastic SRL
	uwire dat_t  bdat;
	for(genvar  i = 0; i < DATA_WIDTH; i++) begin : genSRL
		SRL16E srl (
			.CLK(clk),
			.CE(aload),
			.D(adat[i]),
			.A3(aptr[3]), .A2(aptr[2]), .A1(aptr[1]), .A0(aptr[0]),
			.Q(bdat[i])
		);
	end : genSRL

	// Output Register
	logic  BVld = 0;
	assign	ovld = BVld;
	assign	bload = !BVld || ordy;

	always_ff @(posedge clk) begin
		if(rst)  BVld <= 0;
		else     BVld <= bvld || !bload;
	end

	(* EXTRACT_ENABLE = "true" *)
	dat_t  B = 'x;
	assign	odat = B;

	always_ff @(posedge clk) begin
		if(bload)  B <= bdat;
	end

endmodule : skid
