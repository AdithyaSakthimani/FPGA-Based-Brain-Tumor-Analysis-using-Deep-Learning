`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: Department of Computer Engineering, University of Peradeniya
// Engineer: Mahela Ekanayake, Chaminda Weerasinghe
//
// Module Name: pipeline_controller.v
// Description: Configurable multi-stage biomedical image processing pipeline.
//
//   Mode Register bit assignments (sent as first byte of each tile):
//     [0] - Histogram Equalisation (Contrast_Enhancer)
//     [1] - Gamma Correction       (Gamma_Corrector)
//     [2] - Sobel Edge Detection   (Sobel_Edge_Detector)
//     [3] - Otsu Thresholding      (Otsu_Thresholder)
//     [4] - Morphological Erosion  (Morphological_Processor, mode=0)
//     [5] - Morphological Dilation (Morphological_Processor, mode=1)
//     [7:6] - reserved
//
//   When a stage bit is 0, the stage is bypassed: the skid buffer upstream
//   is connected directly to the skid buffer downstream.
//
//   Pipeline order: HEQ → GAMMA → SOBEL → OTSU → MORPH
//
//   Each stage is separated by a skid buffer for backpressure isolation.
//   The gamma_corrector LUT is loaded via a sideband port (lut_we, lut_addr,
//   lut_data) which the UART_RX drives after receiving the mode byte.
//
// Dependencies: histogram_equalizer.sv, gamma_corrector.sv,
//               axis_sobel_edge_detector.v, otsu_thresholder.sv,
//               axis_morphological_processor.v, skid_buffer.sv
//////////////////////////////////////////////////////////////////////////////////

module pipeline_controller #(
    parameter R_I = 7, C_I = 7, W_I = 8, R_K = 3, C_K = 3,
              W_BUS = R_I * C_I * W_I
)(
    input              clk, rstn,

    // Mode register (latched per tile by uart_rx_bio)
    input  [7:0]       mode_reg,

    // Gamma LUT sideband load port
    input  [7:0]       gamma_lut_addr,
    input  [7:0]       gamma_lut_data,
    input              gamma_lut_we,
    input  [7:0]       gamma_param,

    // Slave AXI-Stream (from UART_RX)
    input  [W_BUS-1:0] s_data,
    input              s_valid,
    output             s_ready,

    // Master AXI-Stream (to UART_TX)
    output [W_BUS-1:0] m_data,
    output             m_valid,
    input              m_ready
);

    // -----------------------------------------------------------------------
    // Stage enable signals
    // -----------------------------------------------------------------------
    wire en_heq   = mode_reg[0];
    wire en_gamma = mode_reg[1];
    wire en_sobel = mode_reg[2];
    wire en_otsu  = mode_reg[3];
    wire en_morph = mode_reg[4] | mode_reg[5];
    wire morph_mode = mode_reg[5];   // 0=erosion, 1=dilation

    // -----------------------------------------------------------------------
    // Inter-stage wires: 6 nodes (input + 5 stage outputs)
    // node[0] = pipeline input
    // node[1] = after HEQ
    // node[2] = after GAMMA
    // node[3] = after SOBEL
    // node[4] = after OTSU
    // node[5] = after MORPH  = pipeline output
    // -----------------------------------------------------------------------
    wire [W_BUS-1:0] nd_data  [0:5];
    wire             nd_valid [0:5];
    wire             nd_ready [0:5];

    assign nd_data [0] = s_data;
    assign nd_valid[0] = s_valid;
    assign s_ready     = nd_ready[0];

    assign m_data  = nd_data [5];
    assign m_valid = nd_valid[5];
    assign nd_ready[5] = m_ready;

    // -----------------------------------------------------------------------
    // Stage 1: Histogram Equalisation
    // -----------------------------------------------------------------------
    wire [W_BUS-1:0] heq_m_data;
    wire             heq_m_valid, heq_m_ready;
    wire             heq_s_ready;

    histogram_equalizer #(.R_I(R_I), .C_I(C_I), .W_I(W_I)) HEQ (
        .clk    (clk), .rstn(rstn),
        .s_data (nd_data[0]),  .s_valid(nd_valid[0] & en_heq),
        .s_ready(heq_s_ready),
        .m_data (heq_m_data),  .m_valid(heq_m_valid),
        .m_ready(heq_m_ready)
    );

    // Bypass mux for HEQ
    assign nd_data [1] = en_heq ? heq_m_data  : nd_data [0];
    assign nd_valid[1] = en_heq ? heq_m_valid : nd_valid[0];
    assign heq_m_ready = en_heq ? nd_ready[1] : 1'b0;
    assign nd_ready[0] = en_heq ? heq_s_ready : nd_ready[1];

    // -----------------------------------------------------------------------
    // Stage 2: Gamma Correction
    // -----------------------------------------------------------------------
    wire [W_BUS-1:0] gam_m_data;
    wire             gam_m_valid, gam_m_ready;
    wire             gam_s_ready;

    gamma_corrector #(.R_I(R_I), .C_I(C_I), .W_I(W_I)) GAMMA (
        .clk         (clk), .rstn(rstn),
        .gamma_param (gamma_param),
        .lut_load    (1'b0),
        .lut_addr_in (gamma_lut_addr),
        .lut_data_in (gamma_lut_data),
        .lut_we      (gamma_lut_we),
        .s_data      (nd_data[1]),  .s_valid(nd_valid[1] & en_gamma),
        .s_ready     (gam_s_ready),
        .m_data      (gam_m_data),  .m_valid(gam_m_valid),
        .m_ready     (gam_m_ready)
    );

    assign nd_data [2] = en_gamma ? gam_m_data  : nd_data [1];
    assign nd_valid[2] = en_gamma ? gam_m_valid : nd_valid[1];
    assign gam_m_ready = en_gamma ? nd_ready[2] : 1'b0;
    assign nd_ready[1] = en_gamma ? gam_s_ready : nd_ready[2];

    // -----------------------------------------------------------------------
    // Stage 3: Sobel Edge Detection
    // -----------------------------------------------------------------------
    wire [W_BUS-1:0] sob_m_data;
    wire             sob_m_valid, sob_m_ready;
    wire             sob_s_ready;

    axis_sobel_edge_detector #(
        .R_I(R_I), .C_I(C_I), .W_I(W_I), .R_K(R_K), .C_K(C_K)
    ) SOBEL (
        .clk                 (clk), .rstn(rstn),
        .s_axis_sobel_valid  (nd_valid[2] & en_sobel),
        .s_axis_sobel_data   (nd_data[2]),
        .s_axis_sobel_ready  (sob_s_ready),
        .m_axis_sobel_ready  (sob_m_ready),
        .m_axis_sobel_valid  (sob_m_valid),
        .m_axis_sobel_data   (sob_m_data)
    );

    assign nd_data [3] = en_sobel ? sob_m_data  : nd_data [2];
    assign nd_valid[3] = en_sobel ? sob_m_valid : nd_valid[2];
    assign sob_m_ready = en_sobel ? nd_ready[3] : 1'b0;
    assign nd_ready[2] = en_sobel ? sob_s_ready : nd_ready[3];

    // -----------------------------------------------------------------------
    // Stage 4: Otsu Thresholding
    // -----------------------------------------------------------------------
    wire [W_BUS-1:0] ots_m_data;
    wire             ots_m_valid, ots_m_ready;
    wire             ots_s_ready;

    otsu_thresholder #(.R_I(R_I), .C_I(C_I), .W_I(W_I)) OTSU (
        .clk    (clk), .rstn(rstn),
        .s_data (nd_data[3]),  .s_valid(nd_valid[3] & en_otsu),
        .s_ready(ots_s_ready),
        .m_data (ots_m_data),  .m_valid(ots_m_valid),
        .m_ready(ots_m_ready)
    );

    assign nd_data [4] = en_otsu ? ots_m_data  : nd_data [3];
    assign nd_valid[4] = en_otsu ? ots_m_valid : nd_valid[3];
    assign ots_m_ready = en_otsu ? nd_ready[4] : 1'b0;
    assign nd_ready[3] = en_otsu ? ots_s_ready : nd_ready[4];

    // -----------------------------------------------------------------------
    // Stage 5: Morphological Processing
    // -----------------------------------------------------------------------
    wire [W_BUS-1:0] mor_m_data;
    wire             mor_m_valid, mor_m_ready;
    wire             mor_s_ready;

    axis_morphological_processor #(
        .R_I(R_I), .C_I(C_I), .W_I(W_I), .R_K(R_K), .C_K(C_K)
    ) MORPH (
        .clk                 (clk), .rstn(rstn),
        .mode                (morph_mode),
        .s_axis_morph_valid  (nd_valid[4] & en_morph),
        .s_axis_morph_data   (nd_data[4]),
        .s_axis_morph_ready  (mor_s_ready),
        .m_axis_morph_ready  (mor_m_ready),
        .m_axis_morph_valid  (mor_m_valid),
        .m_axis_morph_data   (mor_m_data)
    );

    assign nd_data [5] = en_morph ? mor_m_data  : nd_data [4];
    assign nd_valid[5] = en_morph ? mor_m_valid : nd_valid[4];
    assign mor_m_ready = en_morph ? nd_ready[5] : 1'b0;
    assign nd_ready[4] = en_morph ? mor_s_ready : nd_ready[5];

endmodule
