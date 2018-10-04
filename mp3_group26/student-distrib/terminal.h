
#ifndef _TERMINAL_H
#define _TERMINAL_H
#include "keyboard.h"

#include "types.h"
#include "lib.h"

/* jumptable for the stdin operation.*/
file_operation_table_t stdin_jumptable;

/* jumptable for the stdout operation.*/
file_operation_table_t stdout_jumptable;

/* initial the terminal.*/
void terminal_init();

/* open the terminal.*/
int32_t terminal_open(const uint8_t* filename);

/* close the terminal opened.*/
int32_t terminal_close(int32_t fd);

/* read the input from the terminal.*/
int32_t terminal_read(int32_t fd, void* buf, int32_t nbtyes);

/* write the data in the buffer to the terminal.*/
int32_t terminal_write(int32_t fd, const void* buf, int32_t nbytes);


#endif 
