/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

module accuf #(
	shortreal  SCALE = 1.0,	// SCALE == 0.0: accumulate squares
	shortreal  BIAS  = 0.0
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [31:0]  a,
	input	logic  avld,
	input	logic  alst,
	output	logic [31:0]  s,
	output	logic  svld
);

	logic [3:0]  Lst = '0;
	always_ff @(posedge  clk) begin
		if(rst)  Lst <= '0;
		else     Lst <= { Lst[2:0], avld && alst };
	end
	assign	svld = Lst[3];

	// P(s) += B(SCALE == 0.0? a : SCALE) * A(a)
	// @todo  Optimize for SCALE = 1.0f
	uwire [31:0]  b = SCALE == 0.0? a : $shortrealtobits(SCALE);

	// If a BIAS is used, its fed through C.
	localparam bit         HAVE_BIAS = BIAS != 0.0;
	localparam bit [31:0]  BIAS_BITS = $shortrealtobits(BIAS);

	uwire  load_bias;
	if(HAVE_BIAS) begin : genBiasInit
		logic  LdBias = 1;
		always_ff @(posedge clk) begin
			if(rst)  LdBias <= 1;
			else     LdBias <= Lst[1];
		end
		assign	load_bias = LdBias;
	end : genBiasInit

	uwire [1:0]  inv;
	uwire [1:0]  ovf;
	uwire [1:0]  unf;
	always_ff @(posedge clk) begin
		assert(!inv) else $warning("%m generated invalid output.");
		assert(!ovf) else $warning("%m generated an overflow.");
		assert(!unf) else $warning("%m generated an underflow.");
	end
	DSPFP32 #(
		// Feature Control Attributes: Data Path Selection
		.A_FPTYPE("B32"),      // B16, B32
		.A_INPUT("DIRECT"),    // Selects A input source, "DIRECT" (A port) or "CASCADE" (ACIN port)
		.BCASCSEL("B"),        // Selects B cascade out data (B, D).
		.B_D_FPTYPE("B32"),    // B16, B32
		.B_INPUT("DIRECT"),    // Selects B input source, "DIRECT" (B port) or "CASCADE" (BCIN port)
		.PCOUTSEL("FPA"),      // Select PCOUT output cascade of DSPFP32 (FPA, FPM)
		.USE_MULT("MULTIPLY"), // Select multiplier usage (DYNAMIC, MULTIPLY, NONE)

		// Programmable Inversion Attributes: Specifies built-in programmable inversion on specific pins
		.IS_CLK_INVERTED(1'b0),
		.IS_FPINMODE_INVERTED(1'b0),
		.IS_FPOPMODE_INVERTED({ 2'b00, !HAVE_BIAS, 4'b00_00 }),
		.IS_RSTA_INVERTED(1'b0),
		.IS_RSTB_INVERTED(1'b1),
		.IS_RSTC_INVERTED(1'b0),
		.IS_RSTD_INVERTED(1'b0),
		.IS_RSTFPA_INVERTED(1'b0),
		.IS_RSTFPINMODE_INVERTED(1'b0),
		.IS_RSTFPMPIPE_INVERTED(1'b0),
		.IS_RSTFPM_INVERTED(1'b0),
		.IS_RSTFPOPMODE_INVERTED(1'b0),

		// Register Control Attributes: Pipeline Register Configuration
		.ACASCREG(1),                      // Number of pipeline stages between A/ACIN and ACOUT (0-2)
		.AREG(1),                          // Pipeline stages for A (0-2)
		.FPA_PREG(1),                      // Pipeline stages for FPA output (0-1)
		.FPBREG(1),                        // Pipeline stages for B inputs (0-1)
		.FPCREG(0),                        // Pipeline stages for C input (0-3)
		.FPDREG(0),                        // Pipeline stages for D inputs (0-1)
		.FPMPIPEREG(1),                    // Selects the number of FPMPIPE registers (0-1)
		.FPM_PREG(1),                      // Pipeline stages for FPM output (0-1)
		.FPOPMREG(1),                      // Selects the length of the FPOPMODE pipeline (0-3)
		.INMODEREG(0),                     // Selects the number of FPINMODE registers (0-1)
		.RESET_MODE("SYNC")                // Selection of synchronous or asynchronous reset. (ASYNC, SYNC).
	)
	DSPFP32_inst (
		// Cascade outputs: Cascade Ports
		.ACOUT_EXP(),
		.ACOUT_MAN(),
		.ACOUT_SIGN(),
		.BCOUT_EXP(),
		.BCOUT_MAN(),
		.BCOUT_SIGN(),
		.PCOUT(),

		// Data outputs: Data Ports
		.FPM_INVALID(inv[0]),
		.FPM_OVERFLOW(ovf[0]),
		.FPM_UNDERFLOW(unf[0]),
		.FPM_OUT(),
		.FPA_INVALID(inv[1]),
		.FPA_OVERFLOW(ovf[1]),
		.FPA_UNDERFLOW(unf[1]),
		.FPA_OUT(s),
		// Cascade inputs: Cascade Ports
		.ACIN_EXP('x),
		.ACIN_MAN('x),
		.ACIN_SIGN('x),
		.BCIN_EXP('x),
		.BCIN_MAN('x),
		.BCIN_SIGN('x),
		.PCIN('x),
		// Control inputs: Control Inputs/Status Bits
		.CLK(clk),
		.FPINMODE('1),            // 1-bit input: Controls select for B/D input data mux.
		.FPOPMODE({  // add(Lst[3]? 0/C : P, M)
			2'b00,
			HAVE_BIAS? { 1'b1, load_bias, 1'b0 } : { Lst[2], 2'b00 },
			2'b01
		}),
		// Data inputs: Data Ports
		.A_SIGN(a[31]),
		.A_EXP(a[30:23]),
		.A_MAN(a[22:0]),
		.B_SIGN(b[31]),
		.B_EXP(b[30:23]),
		.B_MAN(b[22:0]),
		.C(HAVE_BIAS? BIAS_BITS : 'x),
		.D_EXP('x),
		.D_MAN('x),
		.D_SIGN('x),
		// Reset/Clock Enable inputs: Reset/Clock Enable Inputs
		.ASYNC_RST('0),
		.CEA1('0),
		.CEA2('1),
		.CEB('1),
		.CEC('0),
		.CED('0),
		.CEFPA('1),
		.CEFPINMODE('0),
		.CEFPM('1),
		.CEFPMPIPE('1),
		.CEFPOPMODE('1),

		.RSTA(rst),
		.RSTB(avld),
		.RSTC('0),
		.RSTD('0),
		.RSTFPA(rst),
		.RSTFPINMODE('0),
		.RSTFPM(rst),
		.RSTFPMPIPE(rst),
		.RSTFPOPMODE(rst)
	);

endmodule : accuf
