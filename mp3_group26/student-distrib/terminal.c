
#include "terminal.h"

#define BUFFERLENGTH	128




/* initialize flag as a global variable.*/
int flags;
void terminal_init(){
	
	int i, j;
	flag_first_execute = 1;
	stdin_jumptable.open = &terminal_open;
	stdin_jumptable.close = &terminal_close;
	stdin_jumptable.read = &terminal_read;
	stdin_jumptable.write = &terminal_write;
	
	stdout_jumptable.open = &terminal_open;
	stdout_jumptable.close = &terminal_close;
	stdout_jumptable.read = &terminal_read;
	stdout_jumptable.write = &terminal_write;
	
	for(i = 0;i<6;i++)	phys_addr_available[i]=1;
	
	/* initialize each terminal's info */
	for (i=0; i<3; i++) {
		terminals[i].curr_char_idx = 0;
		for (j=0; j<BUFFERLENGTH; j++) {
			terminals[i].share_buf[j]='\0';
		}
		for (j=0; j<BUFFERLENGTH; j++) {
			terminals[i].typed_buf[j]='\0';
		}
		/////////////////////////////////////////
		terminals[i].buffer_ec_idx = 0;
		terminals[i].buffer_size = 0;
		terminals[i].hello_flag = 0;
		////////////////////////////////////////
		terminals[i].vid_on = 0;
		terminals[i].screen_x=0;
		terminals[i].screen_y=0;
		for (j=0; j<6; j++) {
			terminals[i].phys_mem_idx[j]=-1;
		}
		//virtual memory start at 36MB to store each terminal's video memory
		terminals[i].video_mem = (char *)(VIDEO_MEM + (i+1)*FOUR_KB);
	}
	/* start with the first terminal */
	curr_terminal_idx = 0;
}
/*
 * terminal_open
 *   DESCRIPTION: open the terminal.
 *   INPUTS: filename: a pointer to a filename.
 *   OUTPUTS: none
 *   RETURN VALUE: 0: return 0 to where calls this close function.
 *   SIDE EFFECTS: clear the share_buffer to '/0'.
 */
int32_t terminal_open(const uint8_t* filename) {
	int i;

	/* clear the share_buffer we would use later.*/
	for (i=0; i<BUFFERLENGTH; i++) {
		share_buffer[i]='\0';
	}
	return 0;
}

/*
 * terminal_close
 *   DESCRIPTION: open the terminal.
 *   INPUTS: fd: an int32_t integer.
 *   OUTPUTS: none
 *   RETURN VALUE: 0: return 0 to where calls this close function.
 *   SIDE EFFECTS: none.
 */
int32_t terminal_close(int32_t fd) {
	return 0;
}

/*
 * terminal_read
 *   DESCRIPTION: read the input data from the terminal buffer.
 *   INPUTS: fd: an int32_t integer.
 *			 buf: a buffer that would store the data read from the terminal.
 *			 nbytes: the data length that we want to read from the terminal.
 *   OUTPUTS: none
 *   RETURN VALUE: ret: the length of data we read from the terminal.
 *   SIDE EFFECTS: none.
 */
int32_t terminal_read(int32_t fd, void* buf, int32_t nbytes) {

	/* initialize variables.*/
	int i;
	int ret;
	char * buf_temp;
	buf_temp = buf;

	if (buf_temp == NULL) return -1;
	if (nbytes <0) return -1;
	/* prevent interruption.*/
	//disable_irq(0);
	sti();
	while (!terminals[curr_running_terminal_idx].typing_finish);
		//check critical section
		cli();
		//cli_and_save(flags);
	/* check whether buffer overflow, and set return value to the length we read from the terminal.*/
	if (nbytes>BUFFERLENGTH) {
		ret = BUFFERLENGTH;
	}
	else ret = nbytes;

	/* read data from share_buffer and copy them to the parameter buffer. Once reaches end, the loop breaks.*/
	for (i=0; i<ret; i++) {
		/*if (share_buffer[i]=='\0') {
			break;
		}*/
		buf_temp[i] = terminals[curr_running_terminal_idx].share_buf[i];
	}
	/* set return value to the length we copied.*/
	ret = i;
	terminals[curr_running_terminal_idx].typing_finish = 0;
	
	for (i=0; i<BUFFERLENGTH; i++) {
		terminals[curr_terminal_idx].typed_buf[i] = '\0';
	}
	terminals[curr_terminal_idx].curr_char_idx = 0;

	/* allow interruption after the previous loading.*/
	//restore_flags(flags);
	//enable_irq(0);
	return (int32_t)strlen(buf);
}

/*
 * terminal_write
 *   DESCRIPTION: write the data from input buffer to terminal.
 *   INPUTS: fd: an int32_t integer.
 *			 buf: a buffer that stores the data we want to write to the terminal.
 *			 nbytes: the data length that we want to write to the terminal.
 *   OUTPUTS: the data in buffer to terminal
 *   RETURN VALUE: ret: the length of data we write to the terminal.
 *				   -1: if write to terminal fails, return -1.
 *   SIDE EFFECTS: none.
 */
int32_t terminal_write(int32_t fd, const void* buf, int32_t nbytes) {

	/* initialize variables.*/
	cli();
	disable_irq(0);
	int i;
	int ret;
	temp_curr_running_terminal_idx = curr_running_terminal_idx;
	char * buf_temp;
	buf_temp = (char*)buf;
	/* check if buffer is null or nbytes is out of range, return -1 if it satisfies.*/
	if (buf_temp == NULL) return -1;
	if (nbytes <0) return -1;

	/* write a maximum length of 128 bytes to terminal.*/
	/*if (nbytes>BUFFERLENGTH) {
		ret = BUFFERLENGTH;
	}
	else*/
	ret = nbytes;
	/*if(ret>strlen(buf_temp))
		ret = strlen(buf_temp);*/
	/* write characters in the buffer one by one ot the terminal.
	 * break the loop once reaches the end.*/
	for (i=0; i<ret; i++) {
		/* if there's no more character to be printed, break the loop.*/
		/*if (buf_temp[i] == '\0') {
			break;
		}*/
		/*
		if (buf_temp[i] == '\n') {
			putc('\n');
			break;
		}*/
		/* put character to terminal by calling putc function in lib.c.*/
		if(curr_terminal_idx == temp_curr_running_terminal_idx)
		putc(buf_temp[i]);
		else 
		putc_background(buf_temp[i]);
	}

	/* if the buffer is full, go to the new line by putting a newline character to the terminal.*/
		if(i==BUFFERLENGTH)
			putc('\n');

	/* set return value to the length we write to the terminal.*/
	ret = i;
	enable_irq(0);
	sti();
	return ret;
}
