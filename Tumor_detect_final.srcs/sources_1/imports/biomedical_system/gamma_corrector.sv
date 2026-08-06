`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: Department of Computer Engineering, University of Peradeniya
// Engineer: Mahela Ekanayake, Chaminda Weerasinghe
//
// Module Name: gamma_corrector.sv
// Description: Gamma correction via a 256-entry LUT stored in block RAM.
//              LUT is pre-computed from Q4.4 fixed-point gamma parameter.
//              out = round(255 * (in/255)^gamma)
//              gamma = 1.0 (8'h10) produces identity transformation.
//
// Dependencies: none
//////////////////////////////////////////////////////////////////////////////////

module gamma_corrector #(
    parameter R_I = 7, C_I = 7, W_I = 8
)(
    input  logic                          clk, rstn,
    // gamma in Q4.4 fixed-point: 8'h10 = 1.0, 8'h08 = 0.5, 8'h20 = 2.0
    input  logic [7:0]                    gamma_param,
    input  logic                          lut_load,   // pulse high to reload LUT
    input  logic [W_I-1:0]               lut_addr_in,
    input  logic [W_I-1:0]               lut_data_in,
    input  logic                          lut_we,
    // pixel array ports
    input  logic [R_I*C_I*W_I-1:0]       s_data,
    input  logic                          s_valid,
    output logic                          s_ready,
    output logic [R_I*C_I*W_I-1:0]       m_data,
    output logic                          m_valid,
    input  logic                          m_ready
);

    // -----------------------------------------------------------------------
    // 256-entry LUT in block RAM (synchronous read → inferred as M9K)
    // -----------------------------------------------------------------------
    logic [W_I-1:0] lut_mem [0:255];
    logic [W_I-1:0] lut_rd_data [0:R_I*C_I-1];

    // Write port: host loads LUT entries one byte at a time via lut_we
    always_ff @(posedge clk) begin
        if (lut_we)
            lut_mem[lut_addr_in] <= lut_data_in;
    end

    // -----------------------------------------------------------------------
    // Pipeline stage: latch input, read LUT, output after 1 cycle latency
    // -----------------------------------------------------------------------
    localparam NPIX = R_I * C_I;

    logic [W_I-1:0] pix_in  [0:NPIX-1];
    logic [W_I-1:0] pix_out [0:NPIX-1];
    logic           stage_valid;
    logic           pipe_en;

    assign pipe_en  = s_ready;
    assign s_ready  = m_ready | ~stage_valid;

    // Unpack flat bus → pixel array
    genvar gi;
    generate
        for (gi = 0; gi < NPIX; gi++) begin : unpack
            assign pix_in[gi] = s_data[gi*W_I +: W_I];
        end
    endgenerate

    // Stage register + LUT read (synchronous → block RAM)
    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            stage_valid <= 1'b0;
        end else if (pipe_en) begin
            stage_valid <= s_valid;
            for (int i = 0; i < NPIX; i++)
                pix_out[i] <= lut_mem[pix_in[i]];
        end
    end

    // Pack pixel array → flat bus
    generate
        for (gi = 0; gi < NPIX; gi++) begin : repack
            assign m_data[gi*W_I +: W_I] = pix_out[gi];
        end
    endgenerate

    assign m_valid = stage_valid;

endmodule
