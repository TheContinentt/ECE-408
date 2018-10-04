
#include "system_call_handler.h"

#define ONE				1
#define TWO				2
#define THREE			3
#define FOUR			4
#define SIX				6
#define SEVEN			7
#define EIGHT			8
#define FIFTEEN			15
#define SIXTEEN			16
#define TWENTYFOUR		24
#define THIRTYTWO		32
#define ONETWOEIGHT 	128
#define TWOFIVEFIVE		255
#define FOURMB			0x400000
#define FOURKB			4096
#define EIGHTMB			0x800000
#define EIGHTKB			0x2000
#define VMSTART			0x08048000
#define USER_VIRTUAL_START	0x08000000
#define USER_VIRTUAL_END	0x08400000
#define VID_MAP_VIRTUAL		0x08801000

/* getargs:
 * input: buf: the buffer that holds the chars we want.
 *		  nbytes: the number of byes that we want to copy to the buffer.
 * output: none
 * return value: 0
 * side effect: none.
 */
int32_t getargs(uint8_t* buf, int32_t nbytes){
	
	/*set up critical section*/
	cli();
	int i;
	/*find the current running process*/
	for (i=0; i<SIX; i++) {
		if (terminals[curr_running_terminal_idx].phys_mem_idx[i] == -1) break;
	}
	int curr_halt = terminals[curr_running_terminal_idx].phys_mem_idx[i-ONE];
	/*load the current PCB address*/
	PCB_t* curr_getargs_PCB = (PCB_t*) (EIGHTMB-(curr_halt+ONE)*(EIGHTKB));
	/*return the argument according to the number of bytes given by the user*/
	if (curr_getargs_PCB->arg[0] == NULL) return -1;
	if (nbytes < strlen((char *)curr_getargs_PCB->arg)) {
		strcpy((char *)buf, (char *)curr_getargs_PCB->arg);
	}
	else { strncpy((char *)buf, (char *)curr_getargs_PCB->arg, nbytes); }
	/*clear critical section*/
	sti();
	return 0;
}

/* vidmap:
 * input: screen_start: double pointer of type uint8_t.
 * output: none
 * return value: 0
 * side effect: none.
 */
int32_t vidmap(uint8_t** screen_start){
	int i;
	int curr_vidmap;
	/*check whether the pointer is valid*/
	if (((uint32_t)screen_start<USER_VIRTUAL_START) || ((uint32_t)screen_start>=USER_VIRTUAL_END))
		return -1;
	/*find the current running process*/
	for(i = 0; i<SIX ; i++)
		if(terminals[curr_terminal_idx].phys_mem_idx[i]==-1) break;
	curr_vidmap = terminals[curr_terminal_idx].phys_mem_idx[i-1];
	/*set up the terminal info to show that vidmap is used on this terminal*/
	terminals[curr_terminal_idx].vid_on = 1;
	/*reset page to enable the virtual memory address for user vidmap*/
    set_page_1(THIRTYTWO, EIGHTMB+(curr_vidmap*FOURMB), curr_vidmap+ONE, 1,-1,curr_terminal_idx);
	/*assign the virtual memory address according to the terminal index*/
	(*(screen_start)) = (uint8_t*)(VID_MAP_VIRTUAL+FOURKB*curr_terminal_idx);	
	return 0;  //return virtual address
}

/* set_handler:
 * input: handler_address: pointer of void type.
 *		  signum: int32_t signal number.
 * output: none
 * return value: 0
 * side effect: none.
 */
int32_t set_handler(int32_t signum, void* handler_address){return 0;}

/* sigreturn:
 * input: none.
 * output: none
 * return value: 0
 * side effect: none.
 */
int32_t sigreturn (void){return 0;}

/* halt:
 * Description: In this function, the current task is ended and return its parent task.
 * 				In order to implement this object, we restore the info of parents task and reset cr3
 *				to flush the paging.
 * input: status: input number that offered by the halt system call.
 * output: none
 * return value: 0
 * side effect: halt current program.
 */
int32_t halt(uint8_t status){
	int i;
	int curr_halt;
	int terminal_halt;
	
	cli();
	/* if we halt because of system exception, we reset status to print error information to the terminal.*/
	if(exception_flag ==ONE)
		status = TWOFIVEFIVE;
	
	/* check which task we are now halting.*/
	for (i=0; i<SIX; i++) {
		if (terminals[curr_running_terminal_idx].phys_mem_idx[i] == -1) break;
	}
	curr_halt = terminals[curr_running_terminal_idx].phys_mem_idx[i-ONE];
	terminal_halt = i-1;
	/* load the current PCB address.*/
	PCB_t* curr_halt_PCB = (PCB_t*) (EIGHTMB-(curr_halt+ONE)*(EIGHTKB));
	
	/* if we are not try to halt the shell, go into this branch.*/
	if (curr_halt>2) {
		for (i=0; i<EIGHT; i++) {
			/* reset the all the file descriptor info to prevent potential leak.*/
			(curr_halt_PCB->file_descriptor_tabele[i]).file_operation_table_ptr = NULL;
			(curr_halt_PCB->file_descriptor_tabele[i]).inode = 0;
			(curr_halt_PCB->file_descriptor_tabele[i]).file_position = 0;
			(curr_halt_PCB->file_descriptor_tabele[i]).flags = 0;
		}
	}
	
	/* restore ss0 and esp0 to the parent ss0 and esp0 which we stored before.*/
	tss.ss0 = curr_halt_PCB->parent_ss0;
	tss.esp0 = curr_halt_PCB->parent_esp0;
	
	/* clear the parent PCB information that we used before.*/
	curr_halt_PCB->parent_ss0 = 0;
	curr_halt_PCB->parent_esp0 = 0;
	/* restore the esp and ebp value that we stored before.*/
	uint32_t halt_esp= curr_halt_PCB->parent_esp;
	uint32_t halt_ebp= curr_halt_PCB->parent_ebp;

	/* clear the restored parent esp and ebp value.*/
	curr_halt_PCB->parent_ebp = 0;
	curr_halt_PCB->parent_esp = 0;
	/* make this physical address available.*/
	phys_addr_available[curr_halt] = ONE;
	terminals[curr_running_terminal_idx].phys_mem_idx[terminal_halt]= -1;
	
	terminals[curr_running_terminal_idx].vid_on = 0;
		
	/* flush the TLB by resetting cr3.*/
	asm volatile(
		"movl %0, %%cr3\n"
		:
		:"r"(curr_halt_PCB->parent_PDBR)
	);
	/* finally clear the restored parent PDBR.*/
	curr_halt_PCB->parent_PDBR = 0;
	/* if we try to halt shell, reset its own esp and ebp to restart shell again.*/
	if(curr_halt<3){
	asm volatile(
		"movl %0, %%esp\n"
		"addl $8, %%esp\n"
		:
		:"r"(halt_ebp)
	);
		execute((const uint8_t*)"shell");
	}
	/* else we want to store our parent esp to esp register.*/
	asm volatile(
		"movl %0, %%esp\n"
		:
		:"r"(halt_esp)
	);
	/* Then we want to store our parent ebp to ebp register.*/
	asm (
		"movl %0, %%ebp\n"
		:
		:"r"(halt_ebp),"a"(status)
	);
	/* store exception flag to ebx which we didn't used before.*/
	asm("movl %0, %%ebx\n"
		:
		:"r"(exception_flag)
		:"%eax"
		);
		
		/* then add ebx to eax which should be the return value.*/
	asm("addl %%ebx, %%eax\n"
		:
		:
		);
		
		/* finally jump back to ststem_execute where we called its parent task.*/
	asm  (
		"jmp execute_return\n"
		:
		:
		:"%eax"
		);
	
	return -1;
}
/* execute:
 * Description: In this function, we want to execute the functon which indicated by the input arguments.
 *				In order to that, we need to first check whether the physical address is available, create PCB
 *				and kernel stack and setup paging.
 * input: command: the string contains the file name that we want to execute.
 * output: none
 * return value: -1: if errors happen, return -1.
 *				 %eax: return value offered by executed program.
 * side effect: execute a child program.
 */
int32_t execute (const uint8_t* command) {

	cli();
	/* initialize variables.*/
	if (command == NULL) return -1;
	uint8_t filename[ONETWOEIGHT];
	uint8_t curr_arg[ONETWOEIGHT];		//check!!!!!!!!!!!!!!
	uint8_t instruct_start[FOUR];
	int i;
	int filename_len = 0;
	int arg_exist = 0;
	int curr_idx;
	exception_flag = 0;
	int nonexec_flag = 0;
	int exist;
	int filename_loop = 0;
	int len = strlen((int8_t*)command);
	dentry_t curr_dentry;
	/* array for checking whether the file is executable.*/
	uint8_t buf_elf[FOUR];
	/* array of executable file.*/
	uint8_t elf_magic[FOUR] = {0x7f, 0x45, 0x4c, 0x46}; //the magic numbers to check if the file is executable or not

	enable_irq(0);
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if((strncmp((int8_t*)command,(int8_t*)"hello",strlen((int8_t*)command))==0)||(strncmp((int8_t*)command,(int8_t*)"counter",strlen((int8_t*)command))==0)){
		terminals[curr_terminal_idx].hello_flag = 1;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/*parse the input argument and know the filename we want to execute.*/
	if(len>FIFTEEN)
		filename_loop = FIFTEEN;
	else
		filename_loop = len;
	for (i=0; i<len; i++) {
		if ((command[i] == '\0')||(command[i] == '\n')||(command[i] == ' ')) break;
		filename[i] = command[i];
		filename_len++;
	}
	filename[i] = '\0';
	
	/* parse the rest of command as the argument */
	if (filename_len == len) { arg_exist= 0; }
	else {
		arg_exist = 1;
		for (i=filename_len+1; i<len; i++) {
				curr_arg[i-filename_len-1] = command[i];
		}
		curr_arg[len-filename_len-1] = '\0';
	}
	
	/*Executable check*/
	exist = read_dentry_by_name(filename, &curr_dentry);
	
	/* if not found, return -1.*/
	if (exist == -1) return -1;
	
	/* check whether the file is executable.*/
	read_data(curr_dentry.inode_idx, 0, buf_elf, FOUR);
	for (i=0; i<FOUR; i++) {
		if (buf_elf[i] != elf_magic[i]) {
			nonexec_flag = ONE;
			break;
		}
	}
	
	/* return -1 if not executable.*/
	if (nonexec_flag) return -1;
	
	/* check which pysical address block is available now.*/
	for (i=0; i<SIX; i++) {
		if (phys_addr_available[i]) {
			phys_addr_available[i] = 0;
			break;
		}
	}

	/* return -1 if there's no more physical address available.*/
	if (i>=SIX) return -1;
	curr_idx = i;

	
	for (i=0; i<6; i++) {
		if (terminals[curr_running_terminal_idx].phys_mem_idx[i]==-1) {
			terminals[curr_running_terminal_idx].phys_mem_idx[i] = curr_idx;
			break;
		}
	}
	/*Create PCB*/
	
	/* allocate PCB pointer according to its index in the physical address array.*/
	PCB_t* curr_PCB =(PCB_t*) (EIGHTMB-(curr_idx+1)*(EIGHTKB));
	
	/*Store the parent program's info in order to halt after.*/
	asm volatile (
			"movl	%%ebp, %0\n"
			"movl	%%esp, %1\n"
			:"=r"(curr_PCB->parent_ebp), "=r"(curr_PCB->parent_esp)
			:
	);
	
	

	if (arg_exist == 0) {
		curr_PCB->arg[0] = NULL;
	}
	else {
		strcpy((char *)curr_PCB->arg, (char *)curr_arg);
	}
	
	/* open the stdin and set its initial value.*/
	curr_PCB->file_descriptor_tabele[0].file_operation_table_ptr = &stdin_jumptable;
	curr_PCB->file_descriptor_tabele[0].inode = NULL;
	curr_PCB->file_descriptor_tabele[0].file_position = 0;
	curr_PCB->file_descriptor_tabele[0].flags = ONE; 
	/* open the stdout and set its initial value.*/
	curr_PCB->file_descriptor_tabele[ONE].file_operation_table_ptr = &stdout_jumptable;
	curr_PCB->file_descriptor_tabele[ONE].inode = NULL;
	curr_PCB->file_descriptor_tabele[ONE].file_position = 0;
	curr_PCB->file_descriptor_tabele[ONE].flags = ONE; 
	/* initialize all the other unused file descriptor table elements' value.*/
	for (i=TWO; i<EIGHT; i++) {
		curr_PCB->file_descriptor_tabele[i].file_operation_table_ptr = NULL;
		curr_PCB->file_descriptor_tabele[i].inode = 0;
		curr_PCB->file_descriptor_tabele[i].file_position = 0;
		curr_PCB->file_descriptor_tabele[i].flags = 0;
	}

	
	
		/*save the previous PDBR before reset paging if we are not executing shell.*/
	if (curr_idx > 2) {
		asm volatile(
			"movl %%cr3, %0\n"
			:"=r"(curr_PCB->parent_PDBR)
			:
		);
		curr_PCB->parent_esp0 = tss.esp0;
		curr_PCB->parent_ss0 = tss.ss0;
	}
	/* else reset esp0 and ss0 to reload shell.*/
	else {
		curr_PCB->parent_esp0 = (uint32_t) (EIGHTMB-curr_idx*(EIGHTKB));
		curr_PCB->parent_ss0 = KERNEL_DS;
	}
	
	/*Set paging*/
	int32_t curr_phys_addr = EIGHTMB+curr_idx*FOURMB;
	set_page(THIRTYTWO, curr_phys_addr,curr_idx+ONE, 0,-1);
	/* if curr_idx = 0, flush shell paging.*/
	if (curr_idx<3) {
		asm volatile(
			"movl %%cr3, %0\n"
			:"=r"(curr_PCB->parent_PDBR)
			:
		);
	}
	
	/*Load file to virtual memory*/
	int file_size = get_file_size(curr_dentry.inode_idx);
	read_data(curr_dentry.inode_idx, 0, (uint8_t*)VMSTART, file_size);
	

	
	/*Prepare for execution*/
	tss.ss0 = KERNEL_DS;
	tss.esp0 = (uint32_t) (EIGHTMB-curr_idx*(EIGHTKB));
	/*Calculate instruction start address*/

	read_data(curr_dentry.inode_idx,TWENTYFOUR,instruct_start,FOUR);
	uint32_t addr_asm = (instruct_start[THREE]<<TWENTYFOUR)|(instruct_start[TWO]<<SIXTEEN)|(instruct_start[ONE]<<EIGHT)|instruct_start[0];
	/* push the iret info we need to comeback to the stack and then call the iret to the program we want to execute.*/
	/* 0x083ffffc is the end of the program image in the virtual memory (132MB-4B)*/
	asm volatile (
			"pushl %0\n"
			"pushl $0x083ffffc\n"
			"pushf\n"
			"pushl %1\n"
			"pushl %2\n"
			:
			:"r"(USER_DS), "r"(USER_CS),"r"(addr_asm)
	);
	/*call iret*/
	
	asm volatile (
		"iret \n"
	);
	/* return the assembly linkage.*/
	asm (
		"execute_return:\n"
		"leave\n"
		"ret\n"
		:
		:
		:"%eax"
		);
	return -1;
} 

/* read:
 * Description: read to the file and call the read function used in the file system.
 * input: fd: file descriptor index value.
 *		  buf: the buffer that where we want to store the file data.
 *		  nbytes: the number of bytes that we want to read the data.
 * output: none
 * return value: return value that returned from the read function in the file_system.c.
 * side effect: read a file.
 */
int32_t read(int32_t fd, void* buf, int32_t nbytes){
	int i;
	cli();
	if(fd==1) return -1;
	if (fd<0 || fd >SEVEN) return -1;
	
	/* should return error if call write in the read function or fd index is out of range.*/
	for (i=0; i<SIX; i++) {
		if (terminals[curr_running_terminal_idx].phys_mem_idx[i] == -1) break;
	}
	int curr_halt = terminals[curr_running_terminal_idx].phys_mem_idx[i-ONE];
	/* load the current PCB address.*/
	PCB_t* curr_PCB = (PCB_t*) (EIGHTMB-(curr_halt+ONE)*(EIGHTKB));
	/* if the flag is set, return error.*/
	if (curr_PCB->file_descriptor_tabele[fd].flags == 0) return -1;
	/* call the file read function introduced in file_system.c.*/
	
	return (curr_PCB->file_descriptor_tabele[fd]).file_operation_table_ptr->read((int32_t)(&curr_PCB->file_descriptor_tabele[fd]), buf, nbytes);
}

/* write:
 * Description: write to the file and call the write function used in the file system.
 * input: fd: file descriptor index value.
 *		  buf: the buffer that where we want to store the file data.
 *		  nbytes: the number of bytes that we want to write the data.
 * output: none
 * return value: return value that returned from the write function in the file_system.c.
 * side effect: write a file or write a terminal.
 */
int32_t write(int32_t fd, const void* buf, int32_t nbytes){
	int i;
	cli();
	/* should return error if call read in the write function or fd index is out of range.*/
	if(fd==0) return -1;
	if (fd<0 || fd >SEVEN) return -1;
	/* check which physical address is available now.*/
	for (i=0; i<SIX; i++) {
		if (terminals[curr_running_terminal_idx].phys_mem_idx[i] == -1) break;
	}
	int curr_halt = terminals[curr_running_terminal_idx].phys_mem_idx[i-ONE];
	//terminal_halt = i-1;
	/* load the current PCB address.*/
	PCB_t* curr_PCB = (PCB_t*) (EIGHTMB-(curr_halt+ONE)*(EIGHTKB));
	/* if the flag is set, return error.*/
	if (curr_PCB->file_descriptor_tabele[fd].flags == 0) return -1;
	/* call the file read function introduced in file_system.c.*/
	return (curr_PCB->file_descriptor_tabele[fd]).file_operation_table_ptr->write(fd, buf, nbytes);
}

/* open:
 * Description: open the file.
 * input: filename: the file that we want to open.
 * output: none
 * return value: curr_fd: current fd index.
 * side effect: open the file.
 */
int32_t open(const uint8_t* filename){
	cli();
	int i;
	int ret;
	int curr_fd = -1;
	/* return error if filename is invalid.*/
	if (filename == NULL) return -1;
	/* call the function read_dentry_by_name.*/
	dentry_t curr_dentry_open;
	ret = read_dentry_by_name(filename, &curr_dentry_open);
	if (ret == -1) return -1;
	/* check which physical address is available now.*/
	for (i=0; i<SIX; i++) {
		if (terminals[curr_running_terminal_idx].phys_mem_idx[i] == -1) break;
	}
	int curr_halt = terminals[curr_running_terminal_idx].phys_mem_idx[i-ONE];
	/* load the current PCB address.*/
	PCB_t* curr_PCB = (PCB_t*) (EIGHTMB-(curr_halt+ONE)*(EIGHTKB));
	/* set file descriptor flag from unused to in using.*/
	for (i=0; i<EIGHT; i++) {
		if (curr_PCB->file_descriptor_tabele[i].flags == 0) {
			curr_fd = i;
			break;
		}
	}
	/* return error if no available index found.*/
	if (curr_fd == -1) return -1;
	
	/* open RTC if file type is 0.*/
	if (curr_dentry_open.file_type == 0) {
		/* assign the RTC jumptable to file_operation_table_ptr.*/
		curr_PCB->file_descriptor_tabele[curr_fd].file_operation_table_ptr = &RTC_jumptable; //declare RTC
		curr_PCB->file_descriptor_tabele[curr_fd].inode = NULL;
		curr_PCB->file_descriptor_tabele[curr_fd].file_position = 0;
		/* set current index to be in using.*/
		curr_PCB->file_descriptor_tabele[curr_fd].flags = ONE;
	}
	/* open directory if file type is 1.*/
	else if (curr_dentry_open.file_type == ONE) {
		/* assign the DIR jumptable to file_operation_table_ptr.*/
		curr_PCB->file_descriptor_tabele[curr_fd].file_operation_table_ptr = &DIR_jumptable; //declare DIR
		curr_PCB->file_descriptor_tabele[curr_fd].inode = NULL;
		curr_PCB->file_descriptor_tabele[curr_fd].file_position = 0;
		/* set current index to be in using.*/
		curr_PCB->file_descriptor_tabele[curr_fd].flags = ONE;
	}
	/* open regular file if file type is 2.*/
	else {
		/* assign the FILE jumptable to file_operation_table_ptr.*/
		curr_PCB->file_descriptor_tabele[curr_fd].file_operation_table_ptr = &FILE_jumptable; //declare FILE
		curr_PCB->file_descriptor_tabele[curr_fd].inode = curr_dentry_open.inode_idx;
		curr_PCB->file_descriptor_tabele[curr_fd].file_position = 0;
		/* set current index to be in using.*/
		curr_PCB->file_descriptor_tabele[curr_fd].flags = ONE;
	}
	
	curr_PCB->file_descriptor_tabele[curr_fd].file_operation_table_ptr->open(filename);
	return curr_fd;
}

/* close:
 * Description: close the file.
 * input: fd: the file descriptor number that we want to close and clear all the information.
 * output: none
 * return value: 0: if no errors happen.
 *				 1: if errors happen
 * side effect: close the file.
 */
int32_t close(int32_t fd){
	int i;
	/* return error if fd is invalid.*/
	if (fd<0 || fd >SEVEN) return -1;
	/* set file descriptor flag from unused to in using.*/
	for (i=0; i<SIX; i++) {
		if (terminals[curr_running_terminal_idx].phys_mem_idx[i] == -1) break;
	}
	int curr_halt = terminals[curr_running_terminal_idx].phys_mem_idx[i-ONE];
	/* load the current PCB address.*/
	PCB_t* curr_PCB = (PCB_t*) (EIGHTMB-(curr_halt+ONE)*(EIGHTKB));
	/* clear the file descriptor information.*/
	if ((fd<2)||(curr_PCB->file_descriptor_tabele[fd].flags == 0)) return -1;
	curr_PCB->file_descriptor_tabele[fd].file_operation_table_ptr=NULL;
	curr_PCB->file_descriptor_tabele[fd].inode = 0;
	curr_PCB->file_descriptor_tabele[fd].file_position = 0;
	curr_PCB->file_descriptor_tabele[fd].flags = 0;
	return 0;
}
