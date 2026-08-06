## =============================================================================
## File    : image_enhancement_top.xdc
## Target  : Digilent PYNQ-Z2  (XC7Z020-1CLG400C)
## Top     : image_enhancement_top
## Clock   : 125 MHz onboard oscillator
## =============================================================================

## -----------------------------------------------------------------------------
## Clock  -  125 MHz onboard oscillator  (Pin H16)
## -----------------------------------------------------------------------------
set_property PACKAGE_PIN H16 [get_ports clk]
set_property IOSTANDARD  LVCMOS33 [get_ports clk]
create_clock -add -name sys_clk_pin -period 8.000 -waveform {0 4} [get_ports clk]

## -----------------------------------------------------------------------------
## Reset  -  BTN0  (Pin D19, active HIGH on PYNQ-Z2)
## -----------------------------------------------------------------------------
set_property PACKAGE_PIN D19 [get_ports btn0]
set_property IOSTANDARD  LVCMOS33 [get_ports btn0]

## -----------------------------------------------------------------------------
## LEDs
##   led[0]  LD0  Pin R14  -  Heartbeat (1 Hz blink, confirms clock running)
##   led[1]  LD1  Pin P14  -  DONE (lights when processing is complete)
##   led[2]  LD2  Pin N16  -  s_valid (pipeline input handshake active)
##   led[3]  LD3  Pin M14  -  m_valid (pipeline output ready)
## -----------------------------------------------------------------------------
set_property PACKAGE_PIN R14 [get_ports {led[0]}]
set_property IOSTANDARD  LVCMOS33 [get_ports {led[0]}]

set_property PACKAGE_PIN P14 [get_ports {led[1]}]
set_property IOSTANDARD  LVCMOS33 [get_ports {led[1]}]

set_property PACKAGE_PIN N16 [get_ports {led[2]}]
set_property IOSTANDARD  LVCMOS33 [get_ports {led[2]}]

set_property PACKAGE_PIN M14 [get_ports {led[3]}]
set_property IOSTANDARD  LVCMOS33 [get_ports {led[3]}]

## -----------------------------------------------------------------------------
## Timing exceptions
## -----------------------------------------------------------------------------
set_false_path -from [get_ports btn0]

## -----------------------------------------------------------------------------
## Zynq PL configuration voltage
## -----------------------------------------------------------------------------
set_property CFGBVS        VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3  [current_design]
