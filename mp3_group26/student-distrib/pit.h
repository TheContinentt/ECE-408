#ifndef PIT_H
#define PIT_H

#include "i8259.h"
#include "lib.h"
#include "types.h"
#include "keyboard.h"
#include "system_call_handler.h"

/* function for initialize PIT */
void PIT_init();
/* function for PIT handler */
void PIT_interrupt_handler();

#endif
