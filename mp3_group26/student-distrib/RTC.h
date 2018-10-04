
#ifndef RTC_H
#define RTC_H

#include "i8259.h"
#include "lib.h"
#include "types.h"
#include "keyboard.h"
/*
	Reference: wiki.osdev.org RTC section
*/

/* externt the jumptable for the RTC for the use in system_call.*/
file_operation_table_t RTC_jumptable;

/*initialzie the RTC*/
void RTC_init();

/*handler for RTC interrupt	s*/
void RTC_interrupt_handler();

/* set RTC rate.*/
void RTC_set_rate(int32_t input);

/* open RTC.*/
int32_t RTC_open(const uint8_t* filename);

/* close the RTC.*/
int32_t RTC_close(int32_t fd);

/* write rate to RTC.*/
int32_t RTC_write(int32_t fd, const void* buf, int32_t nbytes);

/* wait for the RTC interrupt.*/
int32_t RTC_read(int32_t fd, void* buf, int32_t nbytes);
#endif
