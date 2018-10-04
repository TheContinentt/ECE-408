
#include "RTC.h"
/*
	Reference: wiki.osdev.org RTC section
*/

#define NMI_A	0x8A
#define NMI_B	0x8B
#define CLEAR_C 0x0C

#define	RTC_PORT_INDEX	0x70
#define RTC_PORT_DATA	0x71

#define HIGH_FOUR_BIT	0xF0
#define BIT_SIX			0x40

#define MAXIMUMF	1024
#define TWO				2
#define ONESIX		16
#define EIGHT       8
#define FOUR        4
#define RTC_RATE    0x02
#define RTC_MASK    0xF

/* set RTC rate to 0x0F */
char rate = RTC_RATE;  //initial value of rate
int32_t interrupt_bool = 0;
/*
 * RTC_init
 *   DESCRIPTION: initialize the RTC, change the corresponding registers to turn on periodic interrupt and set the interrupt rate
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: Initialzie the RTC, update values of register A and B in RTC
 */
void RTC_init() {
	/*turn on the irq[8] line in PIC*/
	enable_irq(EIGHT);
	int freq;
	freq = EIGHT;
	/*fill in the jump table*/
	RTC_jumptable.read = RTC_read;
	RTC_jumptable.write = RTC_write;
	RTC_jumptable.open = RTC_open;
	RTC_jumptable.close = RTC_close;
	
	/*begin the critical section*/
	cli();

	RTC_set_rate(RTC_RATE);  //the initial value of frequency/rate = 2
	/*Turn on IRQ 8*/
	/*select register B, and desable NMI*/
	outb(NMI_B, RTC_PORT_INDEX);
	/*read the current value on register B*/
	char prev_b = inb(RTC_PORT_DATA);
	/*set the index again*/
	outb(NMI_B, RTC_PORT_INDEX);
	/*write the previous value 0Red with 0x40. This turns on bit 6 of register B*/
	outb(prev_b | BIT_SIX, RTC_PORT_DATA);


	/*end the critical section*/
	sti();
}
/*
 * RTC_interrupt_handler
 *   DESCRIPTION: handler for RTC interrupt, called when periodic RTC interrupt recieved, manipulate register C to enable next interrupt signal
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: manipulate register C, enables next interrupt signal
 */
void RTC_interrupt_handler() {

	asm volatile(
		"pushal"
		:
		:);
/*increase counter for loading*/
	loading_counter++;

	/*select register C*/
	outb(CLEAR_C, RTC_PORT_INDEX);
	/*throw away contents*/
	inb(RTC_PORT_DATA);

	interrupt_bool = 1;
	
	/*Send end of interrupt to irq 8*/
	send_eoi(EIGHT);

	asm volatile(
		"popal"
		:
		:);
	
	asm("leave");
	asm("iret");
}

/*
 * RTC_set_rate
 *   DESCRIPTION: set RTC rate according to input rate.
 *   INPUTS: input: the rate we want to set our RTC.
 *   OUTPUTS: none
 *   RETURN VALUE: 0: return 0 to where calls this function.
 *   SIDE EFFECTS: none.
 */
void RTC_set_rate(int32_t input){
	cli();
	/* bitmask input and store value to variable rate.*/
	rate = input&RTC_MASK; //the rate can no larger than 15 
	/*Change the interrupt rate*/
	/*set index to register A, disable NMI*/
	outb(NMI_A, RTC_PORT_INDEX);
	/*get index value of register A*/
	char prev = inb(RTC_PORT_DATA);
	/*reset index to A*/
	outb(NMI_A, RTC_PORT_INDEX);
	/*write only our rate to A*/
	outb((prev & HIGH_FOUR_BIT) | rate, RTC_PORT_DATA);
	sti();
}

/*
 * RTC_read
 *   DESCRIPTION: read RTC.
 *   INPUTS: fd: an int32_t integer.
 *			 buf: a pointer where stores the frequency of RTC we want to set.
 *			 nbytes: the bytes that writes per time.
 *   OUTPUTS: none
 *   RETURN VALUE: 0: return 0 to where calls this function.
 *   SIDE EFFECTS: none.
 */
int32_t RTC_read(int32_t fd, void* buf, int32_t nbytes){
	/*enable interrupts*/
	sti();
	/*disable all other interrupts except RTC*/
	disable_irq(0);
	disable_irq(1);
	/*wait for the RTC interrupt*/
	interrupt_bool = 0;
	while(!interrupt_bool);
	/*enable all interrupts*/
	enable_irq(0);
	enable_irq(1);
	return 0;
}

/*
 * RTC_write
 *   DESCRIPTION: write RTC rate value.
 *   INPUTS: fd: an int32_t integer.
 *			 buf: a pointer where stores the frequency of RTC we want to set.
 *			 nbytes: the bytes that writes per time.
 *   OUTPUTS: none
 *   RETURN VALUE: nbytes: the bytes that writes per time.
 *				   -1: if write RTC frequency fails, return -1.
 *   SIDE EFFECTS: none.
 */
int32_t RTC_write(int32_t fd, const void* buf, int32_t nbytes){
	/* initialize variables.*/
	cli();
	int frequency;
	double temp;
	int temp_t;
	int rate_a;

	/* if buffer is null or frequency is out of range, return -1.*/
	if((buf == NULL)||(nbytes!=FOUR))
		return -1;

	/* read frequency from the input buffer pointer.*/
	frequency = *(int*)buf;
	if((frequency>MAXIMUMF)||((int)frequency!=frequency)||(frequency<0)||(frequency==0))
		return -1;
	temp = frequency;
	rate_a = 0;

	/* divide frequency by 2 each time and calculate the RTC rate.*/
	while(temp!=1){
		temp_t = temp;

		/* if frequency is not divisible by 2, then break out of the loop.*/
		if(temp!=temp_t)
			return -1;
		temp /= TWO;
		rate_a++;
	}

	/* call RTC_set_rate and set the RTC rate.*/
	RTC_set_rate(ONESIX-rate_a);
	sti();
	return nbytes;
}

/*
 * RTC_close
 *   DESCRIPTION: close the RTC
 *   INPUTS: fd: an int32_t integer.
 *   OUTPUTS: none
 *   RETURN VALUE: 0: return 0 to where calls this function.
 *   SIDE EFFECTS: none.
 */
int32_t RTC_close(int32_t fd){
	RTC_set_rate(RTC_RATE);
	return 0;
}

/*
 * RTC_open
 *   DESCRIPTION: open the RTC
 *   INPUTS: filename: an uint8_t pointer to a filename.
 *   OUTPUTS: none
 *   RETURN VALUE: 0: return 0 to where calls this function.
 *   SIDE EFFECTS: none.
 */
int32_t RTC_open(const uint8_t* filename){
	//RTC_set_rate(RTC_RATE);
	return 0;
}
