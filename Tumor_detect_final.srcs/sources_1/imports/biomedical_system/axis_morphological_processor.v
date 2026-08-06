`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: Department of Computer Engineering, University of Peradeniya
// Engineer: Mahela Ekanayake, Chaminda Weerasinghe
//
// Module Name: axis_morphological_processor.v
// Description: AXI-Stream wrapper for morphological_processor.sv.
//              LATENCY = DEPTH + 1 where DEPTH = LEVEL*(LEVEL+1)/2,
//              LEVEL = ceil(log2(R_K*C_K)).
//
// Dependencies: morphological_processor.sv, skid_buffer.sv
//////////////////////////////////////////////////////////////////////////////////

module axis_morphological_processor #(
    parameter R_I = 7, C_I = 7, W_I = 8, R_K = 3, C_K = 3,
              LEVEL   = $clog2(R_K*C_K),
              DEPTH   = LEVEL * (LEVEL+1) / 2,
              LATENCY = DEPTH + 1
)(
    input                       clk, rstn,
    input                       mode,              // 0=erosion, 1=dilation
    output                      s_axis_morph_ready,
    input                       s_axis_morph_valid,
    input  [R_I*C_I*W_I-1:0]   s_axis_morph_data,
    input                       m_axis_morph_ready,
    output                      m_axis_morph_valid,
    output [R_I*C_I*W_I-1:0]   m_axis_morph_data
);

    wire [R_I*C_I*W_I-1:0] i_data;
    wire                    i_ready;

    morphological_processor #(
        .R_I(R_I), .C_I(C_I), .W_I(W_I), .R_K(R_K), .C_K(C_K)
    ) MORPH (
        .clk          (clk),
        .cen          (i_ready),
        .mode         (mode),
        .img_flat     (s_axis_morph_data),
        .final_img_flat(i_data)
    );

    reg [LATENCY-2:0] shift;
    reg               i_valid;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) {i_valid, shift} <= 0;
        else if (i_ready) {i_valid, shift} <= {shift, s_axis_morph_valid};
    end

    skid_buffer #(.WIDTH(R_I*C_I*W_I)) SKID (
        .clk    (clk), .rstn(rstn),
        .s_ready(i_ready),
        .s_valid(i_valid),
        .s_data (i_data),
        .m_ready(m_axis_morph_ready),
        .m_valid(m_axis_morph_valid),
        .m_data (m_axis_morph_data)
    );

    assign s_axis_morph_ready = i_ready;

endmodule
