`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: Department of Computer Engineering, University of Peradeniya
// Engineer: Mahela Ekanayake, Chaminda Weerasinghe
//
// Module Name: axis_sobel_edge_detector.v
// Description: AXI-Stream wrapper for sobel_edge_detector.sv.
//              Follows the same pattern as axis_averaging_filter.v:
//              valid signal is delayed by LATENCY cycles through a shift
//              register to match the pipeline depth of the filter core.
//
// Dependencies: sobel_edge_detector.sv, skid_buffer.sv
//////////////////////////////////////////////////////////////////////////////////

module axis_sobel_edge_detector #(
    parameter R_I = 7, C_I = 7, W_I = 8, R_K = 3, C_K = 3,
              LATENCY = 2   // sobel_edge_detector has 1 FF stage + 1 skid
)(
    input                       clk, rstn,
    output                      s_axis_sobel_ready,
    input                       s_axis_sobel_valid,
    input  [R_I*C_I*W_I-1:0]   s_axis_sobel_data,
    input                       m_axis_sobel_ready,
    output                      m_axis_sobel_valid,
    output [R_I*C_I*W_I-1:0]   m_axis_sobel_data
);

    wire [R_I*C_I*W_I-1:0] i_data;
    wire                    i_ready;

    sobel_edge_detector #(
        .R_I(R_I), .C_I(C_I), .W_I(W_I), .R_K(R_K), .C_K(C_K)
    ) SOBEL (
        .clk       (clk),
        .cen       (i_ready),
        .img_flat  (s_axis_sobel_data),
        .final_img_flat(i_data)
    );

    reg [LATENCY-2:0] shift;
    reg               i_valid;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) {i_valid, shift} <= 0;
        else if (i_ready) {i_valid, shift} <= {shift, s_axis_sobel_valid};
    end

    skid_buffer #(.WIDTH(R_I*C_I*W_I)) SKID (
        .clk    (clk), .rstn(rstn),
        .s_ready(i_ready),
        .s_valid(i_valid),
        .s_data (i_data),
        .m_ready(m_axis_sobel_ready),
        .m_valid(m_axis_sobel_valid),
        .m_data (m_axis_sobel_data)
    );

    assign s_axis_sobel_ready = i_ready;

endmodule
