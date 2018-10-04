#ifndef _SYSTEM_CALL_HANDLER_H
#define _SYSTEM_CALL_HANDLER_H

#include "types.h"
#include "lib.h"
#include "RTC.h"
#include "file_system.h"
#include "page.h"
#include "i8259.h"
#include "terminal.h"
#include "x86_desc.h"
/*in open function of systemcall, put (r.w.open.close) of rtc/file_system/terminal to the jumptable read/write/open/close*/

/* the structure of one entry of the file descriptor.*/
/*typedef struct file_descriptor_t {
	file_operation_table_t* file_operation_table_ptr;
	uint32_t inode;
	uint32_t file_position;
	uint32_t flags;
}file_descriptor_t;*/

/* initialize the exception_flag.*/
int exception_flag;

/* the structure of PCB for each task.*/
typedef struct PCB_t {
	file_descriptor_t file_descriptor_tabele[8];
	uint8_t  arg[1024];		//check!!!!!!!!!!!!!!!!!!!!!!!!!!!
	uint32_t parent_esp0;
	uint16_t parent_ss0;
	uint32_t parent_ebp;
	uint32_t parent_esp;
	uint32_t parent_PDBR;
	uint32_t curr_ebp;
	uint32_t curr_esp;
	uint32_t curr_PDBR;
	uint32_t curr_esp0;
}PCB_t;

/* prototype of getargs.*/
extern int32_t getargs(uint8_t* buf, int32_t nbytes);

/* prototype of vidmap.*/
extern int32_t vidmap(uint8_t** screen_start);

/* prototype of set_handler.*/
extern int32_t set_handler(int32_t signum, void* handler_address);

/* prototype of sigreturn.*/
extern int32_t sigreturn (void);

/* prototype of halt that halt the current .*/
extern int32_t halt(uint8_t status);

/* prototype of execute that execute the command.*/
extern int32_t execute (const uint8_t* command);

/* prototype of read that read the file.*/
extern int32_t read(int32_t fd, void* buf, int32_t nbytes);

/* prototype of write that wrtie to the file.*/
extern int32_t write(int32_t fd, const void* buf, int32_t nbytes);

/* prototype of open that open the files.*/
extern int32_t open(const uint8_t* filename);

/* prototype of close that close the files.*/
extern int32_t close(int32_t fd);

#endif
