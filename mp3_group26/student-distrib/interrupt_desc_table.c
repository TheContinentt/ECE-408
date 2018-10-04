
#include "interrupt_desc_table.h"
/*
	Reference: wiki.osdev.org IDT section
*/
#define SIZE_OF_INT 32
#define SIZE_OF_PIC 8

#define ONE			1
#define SYS_CALL	0x80
#define PIT			32
#define KEYBOARD	33
#define RTC			40
#define MOUSE 		44

/*
 * div_by_zero_error_expt
 *   DESCRIPTION: Handle the divide by zero error exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void div_by_zero_error_expt () {
	printf("Divide-by-zero Error Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * debug_expt
 *   DESCRIPTION: Handle the deebugging exception single-step interrupt, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void debug_expt() {
	printf("Debug Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * NMI_expt
 *   DESCRIPTION: Handle the Non-maskable Interrupt exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void NMI_expt() {
	printf("Non-maskable Interrupt Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * breakpoint_expt
 *   DESCRIPTION: Handle the setting breakpoint exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void breakpoint_expt() {
	printf("Breakpoint Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * overflow_expt
 *   DESCRIPTION: Handle the overflow exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void overflow_expt()  {
	printf("Overflow Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * BR_exceed_expt
 *   DESCRIPTION: Handle the bound range exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void BR_exceed_expt()  {
	printf("Bound Range Exceeded Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * invalid_op_expt
 *   DESCRIPTION: Handle the invalid opcode exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void invalid_op_expt()  {
	printf("Invalide Opcode Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * dev_unavailable_expt
 *   DESCRIPTION: Handle the device not available exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void dev_unavailable_expt()  {
	printf("Device Not Available Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * double_fault_expt
 *   DESCRIPTION: Handle the double fault exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void double_fault_expt()  {
	printf("Double Fault Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * cop_seg_expt
 *   DESCRIPTION: Handle the coprocessor segment overrun exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void cop_seg_expt()  {
	printf("Coprocessor Segment Overrun Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * invalid_tss_expt
 *   DESCRIPTION: Handle the Invalid tss exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void invalid_tss_expt()  {
	printf("Invalid TSS Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * seg_not_present_expt
 *   DESCRIPTION: Handle the segment not present exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void seg_not_present_expt()  {
	printf("Segment Not Present Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * stack_seg_fault_expt
 *   DESCRIPTION: Handle the stack-segment fault exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void stack_seg_fault_expt()  {
	printf("Stack-Segment Fault Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * ge_prot_fault_expt
 *   DESCRIPTION: Handle the general protection fault exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void ge_prot_fault_expt()  {
	printf("General Protection Fault Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * page_fault_expt
 *   DESCRIPTION: Handle the page fault exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void page_fault_expt()  {
	printf("Page Fault Exception Occurs\n");
	//while (1);
	exception_flag = ONE;
	halt(0);
	
}

/*
 * float_ptr_expt
 *   DESCRIPTION: Handle the floating point exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void float_ptr_expt()  {
	printf("x87 Floating-Point Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * align_expt
 *   DESCRIPTION: Handle the alignment check exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void align_expt()  {
	printf("Alignment Check Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * machine_expt
 *   DESCRIPTION: Handle the machine check exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void machine_expt() {
	printf("Machine Check Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * SIMD_expt
 *   DESCRIPTION: Handle the SIMD Floating-Point exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void SIMD_expt() {
	printf("SIMD Floating-Point Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * virt_expt
 *   DESCRIPTION: Handle the Virtualization exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void virt_expt() {
	printf("Virtualization Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * security_expt
 *   DESCRIPTION: Handle the security exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void security_expt() {
	printf("Security Exception Occurs\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * system_call
 *   DESCRIPTION: Handle the system_call exception, print exception line and halt program
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print exception line, halt program
 */
void system_call() {
	printf("system call coming soon...\n");
	exception_flag = ONE;
	halt(0);
	//while(1);
}

/*
 * idt_init
 *   DESCRIPTION: initialize the interrupt description table, set the handled exceptions and PIC descriptors present bit high, present bit of 
 *				  other descriptors low. Fill in the descriptor for handled exceptions(functions declared above), keyboard, RTC and system call
 *			  	  and set all desciptors as 32-bit interrupt gate, DPL as kernel level privilege
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: initialize the IDT, fill in the descriptors and set corresponding present bit
 */


void idt_init() {
	int i;
	/* set up the exceptions in IDT */
	for(i = 0;i<SIZE_OF_INT; i++){
		idt[i].present = 0;
		/* set DPL as kernel level */
		idt[i].dpl = 0;
		/* set storage segment to 0 */
		idt[i].reserved0 = 0;
		/* set type as 0b1110 (32-bit interrupt gate) */
		idt[i].size = 1;
		idt[i].reserved1 = 1;
		idt[i].reserved2 = 1;
		idt[i].reserved3 = 0;
		/* unused byte */
		idt[i].reserved4 = 0;
		/* selector as kernel code segment */
		idt[i].seg_selector = KERNEL_CS;
	}
	
	/* set the offset to each exception handlers, set the present bits high*/
		idt[0].present = 1;
		idt[1].present = 1;
		idt[2].present = 1;
		idt[3].present = 1;
		idt[4].present = 1;
		idt[5].present = 1;
		idt[6].present = 1;
		idt[7].present = 1;
		idt[8].present = 1;
		idt[9].present = 1;
		idt[10].present = 1;
		idt[11].present = 1;
		idt[12].present = 1;
		idt[13].present = 1;
		idt[14].present = 1;
		idt[16].present = 1;
		idt[17].present = 1;
		idt[18].present = 1;
		idt[19].present = 1;
		idt[20].present = 1;
		idt[30].present = 1;
		SET_IDT_ENTRY(idt[0], div_by_zero_error_expt);
		SET_IDT_ENTRY(idt[1], debug_expt);
		SET_IDT_ENTRY(idt[2], NMI_expt);
		SET_IDT_ENTRY(idt[3], breakpoint_expt);
		SET_IDT_ENTRY(idt[4], overflow_expt);
		SET_IDT_ENTRY(idt[5], BR_exceed_expt);
		SET_IDT_ENTRY(idt[6], invalid_op_expt);
		SET_IDT_ENTRY(idt[7], dev_unavailable_expt);
		SET_IDT_ENTRY(idt[8], double_fault_expt);
		SET_IDT_ENTRY(idt[9], cop_seg_expt);
		SET_IDT_ENTRY(idt[10], invalid_tss_expt);
		SET_IDT_ENTRY(idt[11], seg_not_present_expt);
		SET_IDT_ENTRY(idt[12], stack_seg_fault_expt);
		SET_IDT_ENTRY(idt[13], ge_prot_fault_expt);
		SET_IDT_ENTRY(idt[14], page_fault_expt);
		SET_IDT_ENTRY(idt[16], float_ptr_expt);
		SET_IDT_ENTRY(idt[17], align_expt);
		SET_IDT_ENTRY(idt[18], machine_expt);
		SET_IDT_ENTRY(idt[19], SIMD_expt);
		SET_IDT_ENTRY(idt[20], virt_expt);
		SET_IDT_ENTRY(idt[30], security_expt);
		
		
	/* set up the interrupts in IDT */	
	for(i = SIZE_OF_INT; i < NUM_VEC; i++){
		/* multiply 2 since we have a master PIC and a slave PIC */
		if (i<SIZE_OF_INT+SIZE_OF_PIC*2) idt[i].present = 1;
		else idt[i].present = 0;
		/* if not system call, set DPL as kernel level */
		idt[i].dpl = 0;
		/* if system call, set DPL as user level */
		if(i == SYS_CALL) {
			idt[i].present = 1;
			 /* set DPL to user level (3) */
			idt[i].dpl = 3; 
		}
		idt[i].reserved0 = 0;
		/* set type as 0b1110 (32-bit interrupt gate) */
		idt[i].size = 1;
		idt[i].reserved1 = 1;
		idt[i].reserved2 = 1;
		idt[i].reserved3 = 0;
		/* unused byte */
		idt[i].reserved4 = 0x00;
		/* selector as kernel code segment */
		idt[i].seg_selector = KERNEL_CS;
	}	

	/* keyboard */
	/* fill in the offset to keyboard interrupt handler*/
	SET_IDT_ENTRY(idt[KEYBOARD], keyboard_interrupt_handler);
	
	/* RTC */
	/* fill in the offset to RTC interrupt handler*/
	SET_IDT_ENTRY(idt[RTC], RTC_interrupt_handler);
	
	/* PIT */
	/* fill in the offset to PIT interrupt handler*/
	SET_IDT_ENTRY(idt[PIT], PIT_interrupt_handler);
	
	/* MOUSE */
	/* fill in the offset to MOUSE interrupt handler*/
	SET_IDT_ENTRY(idt[MOUSE], mouse_interrupt_handler);
	
	/* system call */
	/* fill in the offset to syscall_handler interrupt handler*/
	SET_IDT_ENTRY(idt[SYS_CALL], syscall_handler);	
	
	/* load IDT into the memory space indicated by idt_desc_ptr(in x86_desc.S)*/	
	lidt(idt_desc_ptr);
	return; 
}
