
/*
	Reference: wiki.osdev.org keyboard section
*/

#include "keyboard.h"

#define KEYBOARD_PORT	0x60
#define BUFFERLENGTH	128
#define ZERO 0
#define ONE 1
#define LAP 0x38
#define LAR 0xB8
#define LSP 0x2A
#define RSP 0x36
#define LSR 0xAA
#define RSR 0xB6
#define LCP 0x1D
#define RCP 0x9D
#define CLP 0x3A
#define F1 0x3B
#define F2 0x3C
#define F3 0x3D
#define BP 0x0E
#define BR 0x8E
#define UPP	0x48  ////////////////////////////
#define THREEB 0x3B
#define THREESEVEN 0x37
#define TWOSIX 0x26
#define MAXIM 0x81
#define THREEEIGHT	0x38


#define TWENTYFOUR			24
#define THIRTYTWO			32
#define TWOFIVEFIVE			255
#define FOURMB			0x400000
#define FOURKB			4096
#define EIGHTMB			0x800000
#define EIGHTKB			0x2000

/*lookup table for scancode input*/
char lower_case_scancodes[] = {'`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', '\b',
					  'T', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', 'E',
					  'C', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '"', '`', '\\',
					  '\\', 'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', 0, 0, 0, ' ', 0, 0, 0, 0, 0, 0, 0};

char upper_case_scancodes[] = {'~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '\b',
					  'T', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '{', '}', 'E',
					  'C', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ':', '"', '~', '|',
					  '|', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', '<', '>', '?', 0, 0, 0, 0, 0, 0, ' ', 0, 0, 0, 0};

char cap_upper_case_scancodes[] = {'`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', '\b',
					  'T', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '[', ']', 'E',
					  'C', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', '"', '`', '\\',
					  '\\', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/', 0, 0, 0, 0, 0, 0, ' ', 0, 0, 0, 0};

char cap_lower_case_scancodes[] = {'~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '\b',
					  'T', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '{', '}', 'E',
					  'C', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ':', '"', '~', '|',
					  '|', 'z', 'x', 'c', 'v', 'b', 'n', 'm', '<', '>', '?', 0, 0, 0, 0, 0, 0, ' ', 0, 0, 0, 0};

/* buffer locally track the typed characters */
char typed_buffer[BUFFERLENGTH];

/* flags to check the current keyboard status */
int la_on = ZERO;
int ra_on = ZERO;
int ls_on = ZERO;
int rs_on = ZERO;
int lctrl_on = ZERO;
int rctrl_on = ZERO;
int cap_on = ZERO;
int backspace_on = ZERO;

/* the current position to update in the buffer */
int curr_index = ZERO;

int temp_counter = 0;


/*
 * keyboard_init
 *   DESCRIPTION: initialze the keyboard, turn on the irq[1] line in PIC
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: irq[1] line in PIC is turned on
 */
void keyboard_init() {
	int i;
	for (i=0; i<3; i++) {
		terminals[i].typing_finish = ZERO;
	}
	enable_irq(1);
}

/*
 * getScancode
 *   DESCRIPTION: get the input from the keyboard
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: character readed
 *   SIDE EFFECTS: irq[1] line in PIC is turned on
 */
char getScancode() {
	char c=ZERO;
	/*read input from port 0x60*/
	c = inb(KEYBOARD_PORT);
	return c;
}

/*
 * clear_buffer
 *   DESCRIPTION: clear the buffer.
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none.
 *   SIDE EFFECTS: clear the buffer and set elements to '\0'.
 */
void clear_buffer() {
	int i;
	for (i=ZERO; i<BUFFERLENGTH; i++) {
		terminals[curr_terminal_idx].typed_buf[i] = '\0';
	}
}

/*
 * keyboard_interrupt_handler
 *   DESCRIPTION: print the readed character on screen
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: character readed
 *   SIDE EFFECTS: irq[1] line in PIC is turned on
 */
void keyboard_interrupt_handler() {
	cli();
    //disable_irq(0);
	nosound();
	/*if (temp_counter % 2 == 0) 
		beep();
	else 
		nosound();
	temp_counter++;*/
	
	asm volatile(
		"pushal\n"
		:
		:);
	/*get input from port 0x60*/
	uint8_t keyboard_flag;
	keyboard_flag = inb(0x64);
	/*if (keyboard_flag &0x20!=0) {
		sti();
		goto FINISH;
	}*/
	uint8_t c = inb(KEYBOARD_PORT);
	int i;
	int ec_buf_temp;  //////////////////////////////////////////////////////////////////////////////
	int ins_flag;
	char curr_character;
	/*if input is a valid input, then print it according to the lookup table defined above*/
	ins_flag = ZERO;
	/* update the status flags */
	switch (c) {
		/* left alt pressed, set flags. */
		case LAP: la_on = ONE; ins_flag = ONE; goto FINISH;
		/* right alt realeased, set flags. */
		case LAR: la_on = ZERO; ins_flag = ONE; goto FINISH;
		/* left shift pressed, set flags. */
		case LSP: ls_on = ONE; ins_flag = ONE; goto FINISH;
		/* right shift pressed, set flags. */
		case RSP: rs_on = ONE; ins_flag = ONE; goto FINISH;
		/* left shift released, set flags. */
		case LSR: ls_on = ZERO; ins_flag = ONE; goto FINISH;
		/* right shift released, set flags. */
		case RSR: rs_on = ZERO; ins_flag = ONE; goto FINISH;
		/* left ctrl pressed, set flags. */
		case LCP: lctrl_on = ONE; ins_flag = ONE; goto FINISH;
		/* right ctrl pressed */
		/* left ctrl released, set flags. */
		case RCP: lctrl_on = ZERO; ins_flag = ONE; goto FINISH;
		/* right ctrl release */
		/* caps_lock pressed, set flags. */
		case CLP: cap_on = cap_on ^ ONE; ins_flag = ONE; goto FINISH;
 		/* backspace pressed, set flags. */
		case BP: backspace_on = ONE; ins_flag = ONE; break;
		/* backspace released, set flags. */
		case BR: backspace_on = ZERO; ins_flag = ONE; goto FINISH;
		/* if right keyboard '*' pressed, skip following process.*/
		default: break;
	}
	
	/*if alt+f2 pressed, change terminal*/
	if((la_on==ONE)&&(c>=F1)&&(c<=F3)){
		/*set up critical section*/
		cli();
		/*black magic to adjust stack*/
		asm volatile(
			"push $0\n"
			"push $0\n"
			"push $0\n"
			:
			:);
		/*check which terminal we are going to display*/
		curr_terminal_idx= (int)c-F1;
		/*memory copy the current terminal backup memory space to the display memory*/
		memcpy((void *)VIDEO_MEM,terminals[curr_terminal_idx].video_mem,(uint32_t)FOURKB);
	
		int pos = terminals[curr_terminal_idx].screen_y*VIDEO_WIDTH+terminals[curr_terminal_idx].screen_x;
		outb(MASK_F,CURSUR_PORTA);
		outb((uint8_t) (pos & MASK_FF),CURSUR_PORTB);
		outb(MASK_E,CURSUR_PORTA);
		outb((uint8_t) ((pos >> MASK_8) & MASK_FF),CURSUR_PORTB);
		*(uint8_t *)((void *)VIDEO_MEM + ((VIDEO_WIDTH * terminals[curr_terminal_idx].screen_y + terminals[curr_terminal_idx].screen_x) << 1) + 1) = 1+curr_terminal_idx;

	
		asm volatile(
			"addl $12, %%esp\n"
			:
			:);
		sti();
		goto FINISH;
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (c == UPP) {
		if(terminals[curr_terminal_idx].buffer_ec_idx<=0)
			goto FINISH;
		terminals[curr_terminal_idx].buffer_ec_idx--;
		ec_buf_temp = terminals[curr_terminal_idx].buffer_ec_idx;
		while (terminals[curr_terminal_idx].curr_char_idx >ZERO) {

			/* call putc function in lib.c to clear elements shows on the screen and clear it in buffer.*/
			putc('\b');
			/* reduce buffer index.*/
			terminals[curr_terminal_idx].curr_char_idx --;
		}
		
		clear_buffer();
		for(i = 0; i < strlen(terminals[curr_terminal_idx].buffer_for_ec[ec_buf_temp]);i++){
			terminals[curr_terminal_idx].typed_buf[i] = terminals[curr_terminal_idx].buffer_for_ec[ec_buf_temp][i];
			putc(terminals[curr_terminal_idx].buffer_for_ec[ec_buf_temp][i]);
		}
		terminals[curr_terminal_idx].curr_char_idx = strlen(terminals[curr_terminal_idx].buffer_for_ec[ec_buf_temp]);
		goto FINISH;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/* Once some scancode which is out of range comes, skip to the FINISH part.*/
	if((!((c>ZERO)&&(c<THREEB)))||(c==THREESEVEN))
		goto FINISH;
	/* when "enter" pressed, send the current typed command */
	if (lower_case_scancodes[c-ONE] == 'E') {
		putc('\n');
		/////////////////////////////////////////////////////////////////////////////////////////
		if(terminals[curr_terminal_idx].buffer_size==BUFFER_SIZE)
			terminals[curr_terminal_idx].buffer_size = 0;
		/////////////////////////////////////////////////////////////////////////////////////////
		/* set current character in typed_buffer to '\n', and copy this buffer to share_buffer.*/
		if(!terminals[curr_terminal_idx].hello_flag){
			terminals[curr_terminal_idx].hello_flag = 0;
		for (i=ZERO; i<BUFFERLENGTH; i++) {
			terminals[curr_terminal_idx].buffer_for_ec[terminals[curr_terminal_idx].buffer_size][i] = terminals[curr_terminal_idx].typed_buf[i];
		}
		terminals[curr_terminal_idx].buffer_size++;
		}
		/////////////////////////////////////////////////////////////////////////////////////////
		terminals[curr_terminal_idx].typed_buf[terminals[curr_terminal_idx].curr_char_idx] = '\n';
		for (i=ZERO; i<BUFFERLENGTH; i++) {
			terminals[curr_terminal_idx].share_buf[i] = terminals[curr_terminal_idx].typed_buf[i];
		}
		
		terminals[curr_terminal_idx].buffer_ec_idx = terminals[curr_terminal_idx].buffer_size; /////////////////////////////////////
		
		/* copy typed_buffer to share_buffer.*/
		terminals[curr_terminal_idx].typing_finish = ONE;
		clear_buffer();

		/* clear current index to the first one of the buffer.*/
		terminals[curr_terminal_idx].curr_char_idx = ZERO;
		//////////////////////////////////////////////////////
		beep();
		//////////////////////////////////////////////////////
		goto FINISH;
	}
	/* when a line is filled up, go to a new line. */
	else if(terminals[curr_terminal_idx].curr_char_idx  == BUFFERLENGTH){
		putc('\n');
		/////////////////////////////////////////////////////////////////////////////////////////
		if(terminals[curr_terminal_idx].buffer_size==BUFFER_SIZE)
			terminals[curr_terminal_idx].buffer_size = 0;
		if(!terminals[curr_terminal_idx].hello_flag){
			terminals[curr_terminal_idx].hello_flag = 0;
		for (i=ZERO; i<BUFFERLENGTH; i++) {
			terminals[curr_terminal_idx].buffer_for_ec[terminals[curr_terminal_idx].buffer_size][i] = terminals[curr_terminal_idx].typed_buf[i];
		}
		terminals[curr_terminal_idx].buffer_size++;
		}
		/////////////////////////////////////////////////////////////////////////////////////////
		/* copy value in typed_buffer to share_buffer.*/
		for (i=ZERO; i<BUFFERLENGTH; i++) {
			terminals[curr_terminal_idx].share_buf[i] = terminals[curr_terminal_idx].typed_buf[i];
		}
		//share_buffer[BUFFERLENGTH] = '\r';
		
		terminals[curr_terminal_idx].buffer_ec_idx = terminals[curr_terminal_idx].buffer_size; /////////////////////////////////////
		
		terminals[curr_terminal_idx].typing_finish = ONE;
		clear_buffer();

		/* reset the buffer index.*/
		terminals[curr_terminal_idx].curr_char_idx  = ZERO;
		if (lower_case_scancodes[c-ONE] == 'E')
			goto FINISH;
		
	}
	/* when "CTRL-L" pressed, clear the screen */
	else if ((lctrl_on) && (c == TWOSIX)) {
		clear();	//clear screen function in lib.h
		clear_buffer();
		terminals[curr_terminal_idx].curr_char_idx  = ZERO;		
		outb(MASK_F,CURSUR_PORTA);
		outb((uint8_t) (0 & MASK_FF),CURSUR_PORTB);
		outb(MASK_E,CURSUR_PORTA);
		outb((uint8_t) ((0 >> MASK_8) & MASK_FF),CURSUR_PORTB);
		*(uint8_t *)((void *)VIDEO_MEM + ((VIDEO_WIDTH * terminals[curr_terminal_idx].screen_y + terminals[curr_terminal_idx].screen_x) << 1) + 1) = 1+curr_terminal_idx;

		goto FINISH;
	}
	/* when backspace is pressed */
	else if (backspace_on) {
		if (terminals[curr_terminal_idx].curr_char_idx >ZERO) {

			/* call putc function in lib.c to clear elements shows on the screen and clear it in buffer.*/
			putc('\b');

			/* reduce buffer index.*/
			terminals[curr_terminal_idx].curr_char_idx --;
		}
		//////////////////////////////////////////////////////
		beep();
		//////////////////////////////////////////////////////
		goto FINISH;
	}
	/* all lower case */
	if ((!cap_on) && (!ls_on )&&(!rs_on)) {

		/* if scancode is in range and no functional button is pressed, print out current character to screen.*/
		if((c<MAXIM)&&(!ins_flag)){

			/* read character according to its scancode.*/
		curr_character = lower_case_scancodes[c-ONE];

		/* add it to buffer and print it.*/
		terminals[curr_terminal_idx].typed_buf[terminals[curr_terminal_idx].curr_char_idx ] = curr_character;
		terminals[curr_terminal_idx].curr_char_idx ++;
		putc(curr_character);
		//////////////////////////////////////////////////////
		beep();
		//////////////////////////////////////////////////////
		}
	}
	/* all upper case */
	else if ((!cap_on) &&(ls_on || rs_on)) {

		/* if shift is pressed, it should print upper number and letters.*/
		if((c<MAXIM)&&(!ins_flag)){

			/* read character according to scancode and print it.*/
		curr_character = upper_case_scancodes[c-ONE];

		/* copy character to buffer.*/
		terminals[curr_terminal_idx].typed_buf[terminals[curr_terminal_idx].curr_char_idx ] = curr_character;
		terminals[curr_terminal_idx].curr_char_idx ++;
		putc(curr_character);
		//////////////////////////////////////////////////////
		beep();
		//////////////////////////////////////////////////////
		}
	}
	/* letter upper case, number lower case */
	else if ((cap_on) && (!ls_on) && (!rs_on)) {

		/* if caps is pressed but shift is not pressed, letter is upper case while number is lower case.*/
		if((c<MAXIM)&&(!ins_flag)){
		curr_character = cap_upper_case_scancodes[c-ONE];

		/* copy character to buffer and print it.*/
		terminals[curr_terminal_idx].typed_buf[terminals[curr_terminal_idx].curr_char_idx ] = curr_character;
		terminals[curr_terminal_idx].curr_char_idx ++;
		putc(curr_character);
		//////////////////////////////////////////////////////
		beep();
		//////////////////////////////////////////////////////
		}
	}
	/* letter lower case, number upper case */
	else if ((cap_on) && (ls_on || rs_on)) {

		/* if caps and shift are both pressed, letter is lower case while number is upper case.*/
		if((c<MAXIM)&&(!ins_flag)){
		curr_character = cap_lower_case_scancodes[c-ONE];

		/* copy character to buffer and print it.*/
		terminals[curr_terminal_idx].typed_buf[terminals[curr_terminal_idx].curr_char_idx ] = curr_character;
		terminals[curr_terminal_idx].curr_char_idx++;
		putc(curr_character);
		//////////////////////////////////////////////////////
		beep();
		//////////////////////////////////////////////////////
		}
	}
FINISH:
	//enable_irq(0);
    sti();
	/*send the end of interrupt signal to irq[1] port*/
	send_eoi(1);
		asm volatile(
		"popal\n"
		:
		:);
	/*return*/
	asm("leave");
	asm("iret");
}
