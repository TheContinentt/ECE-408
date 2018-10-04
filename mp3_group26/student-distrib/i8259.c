/* i8259.c - Functions to interact with the 8259 interrupt controller
 * vim:ts=4 noexpandtab
 */

 /*
	Reference: wiki.osdev.org PIC section
*/
 
#include "i8259.h"
#include "lib.h"

#define SIZE_OF_PIC 8
#define MASK_ALL	0xFF

/* Interrupt masks to determine which interrupts are enabled and disabled */
uint8_t master_mask; /* IRQs 0-7  */
uint8_t slave_mask;  /* IRQs 8-15 */

unsigned int flags;

/*
 * i8259_init
 *   DESCRIPTION: initialzies the PIC by setting the ICW signals of master and slave, turn on the irq[2] port for enabling the slave
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: none
 */
void i8259_init(void) {
	/* clear interrupts and store flags */
	cli_and_save(flags);
	/* mask all interrupts */
	
	//uint8_t cashed_master = master_mask;
	//uint8_t cashed_slave = slave_mask;
	
	master_mask = MASK_ALL;
	slave_mask = MASK_ALL;
	
	uint8_t cashed_master = master_mask;
	uint8_t cashed_slave = slave_mask;	
	

	outb(master_mask, MASTER_8259_PORT+1);
	outb(slave_mask, SLAVE_8259_PORT+1);
	
	/* initialize master */
	outb(ICW1, MASTER_8259_PORT);
	outb(ICW2_MASTER, MASTER_8259_PORT+1);
	outb(ICW3_MASTER, MASTER_8259_PORT+1);
	outb(ICW4, MASTER_8259_PORT+1);
	
	/* initialize slave */
	outb(ICW1, SLAVE_8259_PORT);
	outb(ICW2_SLAVE, SLAVE_8259_PORT+1);
	outb(ICW3_SLAVE, SLAVE_8259_PORT+1);
	outb(ICW4, SLAVE_8259_PORT+1);
	
	
	outb(cashed_master, MASTER_8259_PORT+1);
	outb(cashed_slave, SLAVE_8259_PORT+1);
	
	/* enable the slave connected to IRQ 2 on master PIC */
	enable_irq(2);
	
	/* restore flags and enable interrupts */
	restore_flags(flags);
	sti();
}
/*
 * enable_irq
 *   DESCRIPTION: Enable (unmask) the specified IRQ
 *   INPUTS: number of irq to enable
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: corresponding irq enabled
 */
void enable_irq(uint32_t irq_num) {
	uint16_t port;
	uint8_t value;
	/* If the input number is in master PIC*/
	if (irq_num < SIZE_OF_PIC) {
		/* calculates the port and stores the flipped flag into value and master_mask*/
		port = MASTER_8259_PORT+1;
		value = inb(port) & (~(1 <<  irq_num));
		master_mask = value;
	}
	/* If the input number is in slave PIC*/
	else {
		/* calculates the port and stores the flipped flag into value and slave_mask*/
		port = SLAVE_8259_PORT+1;
		irq_num = irq_num-SIZE_OF_PIC;
		value = inb(port) & (~(1 <<  irq_num));
		slave_mask = value;
	}
	/* sets the flag into the corresponding port*/
	outb(value, port);
}
/*
 * disable_irq
 *   DESCRIPTION: Disable (mask) the specified IRQ
 *   INPUTS: number of irq to disable
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: corresponding irq disabled
 */
void disable_irq(uint32_t irq_num) {
	uint16_t port;
	uint8_t value;
	/* If the input number is in master PIC*/
	if (irq_num < SIZE_OF_PIC) {
		/* calculates the port and stores the flipped flag into value and master_mask*/
		port = MASTER_8259_PORT+1;
		value = inb(port) | (1 << irq_num);
		master_mask = value;
	}
	/* If the input number is in slave PIC*/
	else {
		/* calculates the port and stores the flipped flag into value and slave_mask*/
		port = SLAVE_8259_PORT+1;
		irq_num = irq_num-SIZE_OF_PIC;
		value = inb(port) | (1 << irq_num);
		slave_mask = value;
	}
	outb(value, port);
}
/*
 * send_eoi
 *   DESCRIPTION: Send end-of-interrupt signal for the specified IRQ
 *   INPUTS: number of irq to send end of interrupt signal
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: none
 */
void send_eoi(uint32_t irq_num) {
		cli();
		asm volatile(
		"pushal\n"
		:
		:);
	/* when send EOI to slave PIC */
	if (irq_num >= SIZE_OF_PIC) {
		irq_num = irq_num-SIZE_OF_PIC;
		/* slave PIC is connected to IRQ 2 on master PIC */
		outb(2|EOI, MASTER_8259_PORT);
		outb(irq_num|EOI, SLAVE_8259_PORT);
	}
	/* when send EOI to master PIC */
	else {
		outb(irq_num|EOI, MASTER_8259_PORT);
	}
		asm volatile(
		"popal\n"
		:
		:);
		sti();
}
