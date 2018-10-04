#include "pit.h"
/*
	Reference: wiki.osdev.org PIT section
*/

#define ONE			1
#define TWO 		2
#define THREE		3
#define SIX 		6

#define PIT_CONST 	1193182
#define FREQ 		20
#define SETUP_FLAG	0x34
#define SET_PORT	0x43
#define DATA_PORT	0x40
#define LOW			0xFF

#define FOURKB		4096
#define EIGHT		0x8
#define EIGHTMB		0x800000
#define EIGHTKB		0x2000

#define VID_START	0x000B8000

/*
 * PIT_init
 *   DESCRIPTION: initialize the PIT, change the corresponding registers to turn on periodic interrupt and set the interrupt rate
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: none
 */
void PIT_init(){
	/* enable PIT port on PIC */
	enable_irq(0);
	/* set up critical section */
	cli();
	/* set up default order of three terminals */
	order[0] = 0;
	order[ONE] = ONE;
	order[TWO] = TWO;
	/* set up the default terminal to the first terminal */
	curr_running_terminal_idx = 0;
	/* set up the constant for initialization */
	int reload_val = PIT_CONST/FREQ;
	outb(SETUP_FLAG, SET_PORT);
	outb(reload_val&LOW, DATA_PORT);
	outb(reload_val>>EIGHT, DATA_PORT);
	sti();
}

/* flags for checking whether we have finished opening three shells */
int flag = 0;
int init_flag = 0;
/*
 * PIT_interrupt_handler
 *   DESCRIPTION: handler for PIT interrupt, called when periodic PIT interrupt recieved
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: none
 */
void PIT_interrupt_handler(){
	
	PCB_t* curr_PCB;
	cli();	
	int i = 0;
	int phys_address;
	/* when finish starting three shells */
	if(init_flag){
		init_flag=0;
	}
	if(flag_first_execute){
	flag_first_execute = 0;
	clear();
	send_eoi(0);
	execute((const uint8_t*)"shell"); 
	}
	/* find the current active task (except shell) on the current terminal */
	for(i = ONE; i<SIX; i++)
		if(terminals[curr_running_terminal_idx].phys_mem_idx[i]!=-1)
			break;
	
	/* if no current active task and no shell (check flag), start execute shell */
	if((i==SIX)&&(!flag)){
		/* if we are on the third terminal, which means we have already opened three shells */
		if(curr_running_terminal_idx == TWO)  {
			flag = ONE;
			init_flag = ONE;
		}
		/* find the current process's physical address */
		phys_address = terminals[curr_running_terminal_idx].phys_mem_idx[0];
		curr_PCB =(PCB_t*) (EIGHTMB-(phys_address+ONE)*(EIGHTKB));
		/* store the current stack frame pointers, PDBR and esp0 */
		asm volatile(
			"movl %%esp, %0\n"
			"movl %%ebp, %1\n"
			"movl %%cr3, %2\n"
			:"=r"(curr_PCB->curr_esp),"=r"(curr_PCB->curr_ebp),"=r"(curr_PCB->curr_PDBR)
		);
		curr_PCB->curr_esp0 = tss.esp0;
	

		/*update the current running terminal index*/
		curr_running_terminal_idx++;
		/*if the current running terminal is not the third one, start another shell*/
		if(curr_running_terminal_idx<THREE) {
	/*Send end of interrupt to irq 0*/
			send_eoi(0);
			sti();
			execute((const uint8_t*)"shell"); 
		}
		/*if the current running terminal is the third one, reset the stack frame pointers and PDBR */
		else {
			curr_running_terminal_idx=0;
			curr_PCB = (PCB_t*) (EIGHTMB-EIGHTKB);
			tss.esp0=curr_PCB->curr_esp0;
			asm volatile(
				"movl %0, %%cr3\n"
				:
				:"r"(curr_PCB->curr_PDBR)
			);
			asm volatile(
				"movl %0, %%esp\n"
				"movl %1, %%ebp\n"
				:
				:"r"(curr_PCB->curr_esp),"r"(curr_PCB->curr_ebp)
			);	
		/*Send end of interrupt to irq 0*/
			send_eoi(0);
			sti();
		}
	}
	
	/* if all three shells have been opened */
	else {
		/*if the current running terminal is the one the monitor displays and the terminal is running a program using vid_map, copy the video memory to the screen*/
		if((curr_running_terminal_idx == curr_terminal_idx)&&(terminals[curr_terminal_idx].vid_on==1))
			memcpy((void *)VID_START,terminals[curr_terminal_idx].video_mem,(uint32_t)FOURKB);
		
		/*find the current running terminal's process*/
		for (i=0; i<SIX; i++) {
			if (terminals[curr_running_terminal_idx].phys_mem_idx[i] == -1) break;
		}
		phys_address = terminals[curr_running_terminal_idx].phys_mem_idx[i-1];
		curr_PCB = (PCB_t*) (EIGHTMB-(phys_address+1)*(EIGHTKB));
		
		/*store the current running process's stack frame pointers, PDBR and esp0*/
			asm volatile(
				"movl %%esp, %0\n"
				"movl %%ebp, %1\n"
				"movl %%cr3, %2\n"
				:"=r"(curr_PCB->curr_esp),"=r"(curr_PCB->curr_ebp),"=r"(curr_PCB->curr_PDBR)
			);
		curr_PCB->curr_esp0 = tss.esp0;
		
		/* then switch to the next terminal's active process */
		curr_running_terminal_idx = (curr_running_terminal_idx+1)% THREE;
		
		/*find the new current running process*/
		for (i=0; i<SIX; i++) {
			if (terminals[curr_running_terminal_idx].phys_mem_idx[i] == -1) break;
		}
		phys_address = terminals[curr_running_terminal_idx].phys_mem_idx[i-1];
		curr_PCB = (PCB_t*) (EIGHTMB-(phys_address+1)*(EIGHTKB));
		
		/*set up the stack frame pointers, PDBR and esp0 to the new current running process*/
		tss.esp0=curr_PCB->curr_esp0;
		asm volatile(
			"movl %0, %%cr3\n"
			:
			:"r"(curr_PCB->curr_PDBR)
		);
		send_eoi(0);
		asm volatile(
			"movl %0, %%esp\n"
			"movl %1, %%ebp\n"
			:
			:"r"(curr_PCB->curr_esp),"r"(curr_PCB->curr_ebp)
		);
		sti();
	}
    asm volatile(
		"leave;"
		"iret;"
	);

}
