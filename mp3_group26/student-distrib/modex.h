#ifndef MODEX_H
#define MODEX_H
#include "lib.h"
#define VID_MEM_SIZE        131072
#define MODE_X_MEM_SIZE      65536
#define NUM_SEQUENCER_REGS       5
#define NUM_CRTC_REGS           25
#define NUM_GRAPHICS_REGS        9
#define NUM_ATTR_REGS           22
#define BAR_SIZE_PLANE				1440
void set_text_mode_3(int clear_scr);
int set_mode_X();
#endif
