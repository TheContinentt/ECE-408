#ifndef _SOUND_H
#define _SOUND_H

#include "i8259.h"
#include "lib.h"
#include "types.h"
#include "RTC.h"
#include "page.h"

extern void play_sound(uint32_t nFrequence);
 
extern void nosound();

extern void beep();

extern void beep_init();
 
#endif
