
#ifndef _MOUSE_H
#define _MOUSE_H

#include "i8259.h"
#include "lib.h"
#include "types.h"
#include "lib.h"
void mouse_init();

void mouse_wait(uint8_t a_type);

void mouse_write(uint8_t a_write);

uint8_t mouse_read();

extern void mouse_interrupt_handler();

#endif
