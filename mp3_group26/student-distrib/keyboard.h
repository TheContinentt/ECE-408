


#ifndef _KEYBOARD_H
#define _KEYBOARD_H

#include "i8259.h"
#include "lib.h"
#include "types.h"
#include "RTC.h"
#include "page.h"
#include "sound.h"
#define BUF_SIZE	128

/* buffer shared with system call read function */
char share_buffer[BUF_SIZE];

/* flag shared with system call read function to indicate command enter */
int typing_finish;

/*initialzie the keyboard*/
void keyboard_init();

/*read print input*/
char getScancode();

/*keyboard interrupt handler*/
extern void keyboard_interrupt_handler();

#endif
