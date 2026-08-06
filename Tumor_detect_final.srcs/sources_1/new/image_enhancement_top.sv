`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name : image_enhancement_top.sv
// Target      : Digilent PYNQ-Z2  (XC7Z020-1CLG400C)
// Description : Standalone image-enhancement pipeline.
//               No UART, no AXI, no PS involvement.
//
//   How it works
//   ─────────────
//   A small FSM reads a 7×7 pixel tile from an initialised BRAM (image_rom),
//   drives the pipeline_controller, then writes the processed tile into a
//   second BRAM (result_ram).  All of this happens automatically on power-up
//   once btn0 is released (active-HIGH reset on PYNQ-Z2).
//
//   Observing results
//   ─────────────────
//   • In Vivado Hardware Manager: add ILA on result_we / result_addr /
//     result_data to capture every byte written to result_ram.
//   • LEDs show heartbeat and pipeline-done status.
//
//   Changing the image tile
//   ───────────────────────
//   Edit IMAGE_ROM_INIT below (49 hex bytes = one 7×7 grayscale tile).
//   Re-run synthesis + implementation to bake new pixels into the bitstream.
//
//   Changing the processing mode
//   ─────────────────────────────
//   Edit MODE_REG:
//     bit[0] = Histogram Equalisation
//     bit[1] = Gamma Correction
//     bit[2] = Sobel Edge Detection
//     bit[3] = Otsu Thresholding
//     bit[4] = Morphological Erosion
//     bit[5] = Morphological Dilation
//   Example: 8'h0F = HEQ + GAMMA + SOBEL + OTSU all enabled
//
// Dependencies: pipeline_controller.v, histogram_equalizer.sv,
//               gamma_corrector.sv, axis_sobel_edge_detector.v,
//               sobel_edge_detector.sv, otsu_thresholder.sv,
//               axis_morphological_processor.v, morphological_processor.sv,
//               skid_buffer.sv
//////////////////////////////////////////////////////////////////////////////////

module image_enhancement_top #(
    // ── Image dimensions ───────────────────────────────────────────────────
    parameter integer R_I  = 7,   // tile rows
    parameter integer C_I  = 7,   // tile columns
    parameter integer W_I  = 8,   // bits per pixel
    parameter integer R_K  = 3,   // kernel rows
    parameter integer C_K  = 3,   // kernel columns

    // ── Processing mode (hardwired at synthesis time) ──────────────────────
    // bit[0]=HEQ  bit[1]=GAMMA  bit[2]=SOBEL  bit[3]=OTSU
    // bit[4]=EROSION  bit[5]=DILATION
    parameter [7:0]   MODE_REG    = 8'h0D,   // HEQ + SOBEL + OTSU

    // ── Gamma parameter Q4.4: 8'h10=1.0, 8'h08=0.5, 8'h20=2.0 ────────────
    parameter [7:0]   GAMMA_PARAM = 8'h10    // identity (used when GAMMA enabled)
)(
    input  logic        clk,    // 125 MHz  - PYNQ-Z2 pin H16
    input  logic        btn0,   // BTN0     - PYNQ-Z2 pin D19 (active HIGH = reset)
    output logic [3:0]  led     // LD0..LD3 - debug
);

    // -----------------------------------------------------------------------
    // Derived parameters
    // -----------------------------------------------------------------------
    localparam integer NPIX  = R_I * C_I;          // 49
    localparam integer W_BUS = NPIX * W_I;          // 392

    // -----------------------------------------------------------------------
    // Reset: BTN0 is active-HIGH; pipeline uses active-LOW rstn
    // -----------------------------------------------------------------------
    logic rstn;
    assign rstn = ~btn0;

    // -----------------------------------------------------------------------
    // Heartbeat LED - blinks ~1 Hz to confirm clock is running
    // -----------------------------------------------------------------------
    logic [26:0] hb;
    always_ff @(posedge clk or negedge rstn)
        if (!rstn) hb <= '0;
        else       hb <= hb + 1;
    assign led[0] = hb[26];

    // -----------------------------------------------------------------------
    // IMAGE ROM - initialised with a sample 7×7 grayscale tile
    // Replace these 49 bytes with your actual tumour tile pixels.
    // Values are row-major: pix[0][0], pix[0][1], … pix[6][6]
    // -----------------------------------------------------------------------
    logic [W_I-1:0] image_rom [0:NPIX-1];
    initial begin
        // ── Sample 7×7 tile (edit to use real tumour image data) ───────────
        image_rom[ 0]=8'h52; image_rom[ 1]=8'h55; image_rom[ 2]=8'h60;
        image_rom[ 3]=8'h6A; image_rom[ 4]=8'h6E; image_rom[ 5]=8'h6B;
        image_rom[ 6]=8'h5F;
        image_rom[ 7]=8'h5A; image_rom[ 8]=8'h5E; image_rom[ 9]=8'h70;
        image_rom[10]=8'h7C; image_rom[11]=8'h7D; image_rom[12]=8'h74;
        image_rom[13]=8'h62;
        image_rom[14]=8'h62; image_rom[15]=8'h6A; image_rom[16]=8'h80;
        image_rom[17]=8'h90; image_rom[18]=8'h8E; image_rom[19]=8'h80;
        image_rom[20]=8'h68;
        image_rom[21]=8'h60; image_rom[22]=8'h6E; image_rom[23]=8'h88;
        image_rom[24]=8'h9A; image_rom[25]=8'h98; image_rom[26]=8'h86;
        image_rom[27]=8'h6C;
        image_rom[28]=8'h5C; image_rom[29]=8'h6A; image_rom[30]=8'h7E;
        image_rom[31]=8'h90; image_rom[32]=8'h8E; image_rom[33]=8'h7C;
        image_rom[34]=8'h62;
        image_rom[35]=8'h50; image_rom[36]=8'h5C; image_rom[37]=8'h6C;
        image_rom[38]=8'h7A; image_rom[39]=8'h78; image_rom[40]=8'h6A;
        image_rom[41]=8'h54;
        image_rom[42]=8'h46; image_rom[43]=8'h50; image_rom[44]=8'h5C;
        image_rom[45]=8'h64; image_rom[46]=8'h62; image_rom[47]=8'h56;
        image_rom[48]=8'h44;
    end

    // -----------------------------------------------------------------------
    // RESULT RAM - stores the processed 7×7 tile output
    // Readable via ILA in Hardware Manager
    // -----------------------------------------------------------------------
    (* keep = "true" *) logic [W_I-1:0] result_ram [0:NPIX-1];
    (* keep = "true" *) logic           result_we;
    (* keep = "true" *) logic [5:0]     result_addr;   // 0..48
    (* keep = "true" *) logic [W_I-1:0] result_data;

    always_ff @(posedge clk)
        if (result_we)
            result_ram[result_addr] <= result_data;

    // -----------------------------------------------------------------------
    // Pipeline interface wires
    // -----------------------------------------------------------------------
    logic [W_BUS-1:0] pipe_s_data;
    logic             pipe_s_valid;
    logic             pipe_s_ready;
    logic [W_BUS-1:0] pipe_m_data;
    logic             pipe_m_valid;
    logic             pipe_m_ready;

    // Gamma LUT sideband - tied off (LUT not loaded at runtime in this design)
    // The gamma_corrector will use whatever default LUT values it has (all 0).
    // If you want correct gamma correction, either enable BRAM init below or
    // disable the GAMMA stage in MODE_REG.
    logic [7:0] gamma_lut_addr = 8'h00;
    logic [7:0] gamma_lut_data = 8'h00;
    logic       gamma_lut_we   = 1'b0;

    // -----------------------------------------------------------------------
    // Pipeline controller (the full enhancement chain)
    // -----------------------------------------------------------------------
    pipeline_controller #(
        .R_I(R_I), .C_I(C_I), .W_I(W_I), .R_K(R_K), .C_K(C_K)
    ) PIPELINE (
        .clk           (clk),
        .rstn          (rstn),
        .mode_reg      (MODE_REG),
        .gamma_lut_addr(gamma_lut_addr),
        .gamma_lut_data(gamma_lut_data),
        .gamma_lut_we  (gamma_lut_we),
        .gamma_param   (GAMMA_PARAM),
        .s_data        (pipe_s_data),
        .s_valid       (pipe_s_valid),
        .s_ready       (pipe_s_ready),
        .m_data        (pipe_m_data),
        .m_valid       (pipe_m_valid),
        .m_ready       (pipe_m_ready)
    );

    // -----------------------------------------------------------------------
    // Top-level FSM
    //
    //  LOAD   : Pack image_rom into the flat 392-bit bus, assert s_valid
    //  WAIT   : Hold until pipeline accepts the tile (s_ready handshake)
    //  DRAIN  : Wait for m_valid (pipeline done)
    //  STORE  : Unpack m_data into result_ram one byte per cycle
    //  DONE   : Stay here; LED[1] lights up; ILA can capture result_ram
    // -----------------------------------------------------------------------
    typedef enum logic [2:0] {
        ST_LOAD, ST_WAIT, ST_DRAIN, ST_STORE, ST_DONE
    } fsm_t;

    fsm_t           fsm;
    logic [5:0]     store_cnt;   // 0..48

    // Pack image_rom → flat bus (combinatorial)
    always_comb begin
        pipe_s_data = '0;
        for (int i = 0; i < NPIX; i++)
            pipe_s_data[i*W_I +: W_I] = image_rom[i];
    end

    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            fsm          <= ST_LOAD;
            pipe_s_valid <= 1'b0;
            pipe_m_ready <= 1'b0;
            result_we    <= 1'b0;
            result_addr  <= '0;
            result_data  <= '0;
            store_cnt    <= '0;
        end else begin
            result_we <= 1'b0;   // default: no write

            case (fsm)

                // ── Present tile to pipeline ───────────────────────────────
                ST_LOAD: begin
                    pipe_s_valid <= 1'b1;
                    pipe_m_ready <= 1'b0;
                    if (pipe_s_valid && pipe_s_ready) begin
                        pipe_s_valid <= 1'b0;
                        pipe_m_ready <= 1'b1;
                        fsm          <= ST_DRAIN;
                    end
                end

                // ── Wait for pipeline to finish ────────────────────────────
                ST_DRAIN: begin
                    if (pipe_m_valid && pipe_m_ready) begin
                        pipe_m_ready <= 1'b0;
                        store_cnt    <= '0;
                        fsm          <= ST_STORE;
                    end
                end

                // ── Write result bytes to result_ram one per cycle ─────────
                ST_STORE: begin
                    result_we   <= 1'b1;
                    result_addr <= store_cnt;
                    result_data <= pipe_m_data[store_cnt*W_I +: W_I];
                    if (store_cnt == NPIX-1) begin
                        result_we <= 1'b0;
                        fsm       <= ST_DONE;
                    end else begin
                        store_cnt <= store_cnt + 1;
                    end
                end

                // ── Processing complete ────────────────────────────────────
                ST_DONE: begin
                    // Stay here permanently.
                    // LED[1] is lit.
                    // Use ILA probe on result_ram / result_we to read output.
                end

                default: fsm <= ST_LOAD;
            endcase
        end
    end

    // -----------------------------------------------------------------------
    // Status LEDs
    // -----------------------------------------------------------------------
    assign led[1] = (fsm == ST_DONE);   // lights when processing is complete
    assign led[2] = pipe_s_valid;        // lights while sending to pipeline
    assign led[3] = pipe_m_valid;        // lights when pipeline has output

endmodule