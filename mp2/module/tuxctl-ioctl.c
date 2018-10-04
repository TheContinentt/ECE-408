/* tuxctl-ioctl.c
 *
 * Driver (skeleton) for the mp2 tuxcontrollers for ECE391 at UIUC.
 *
 * Mark Murphy 2006
 * Andrew Ofisher 2007
 * Steve Lumetta 12-13 Sep 2009
 * Puskar Naha 2013
 */

#include <asm/current.h>
#include <asm/uaccess.h>

#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/sched.h>
#include <linux/file.h>
#include <linux/miscdevice.h>
#include <linux/kdev_t.h>
#include <linux/tty.h>
#include <linux/spinlock.h>

#include "tuxctl-ld.h"
#include "tuxctl-ioctl.h"
#include "mtcp.h"

#define debug(str, ...) \
	printk(KERN_DEBUG "%s: " str, __FUNCTION__, ## __VA_ARGS__)
		
#define SHOWZERO	0xE7		/*seven segment of 0.*/
#define SHOWONE		0x06		/*seven segment of 1.*/
#define SHOWTWO		0xCB		/*seven segment of 2.*/
#define SHOWTHREE	0x8F		/*seven segment of 3.*/
#define SHOWFOUR	0x2E		/*seven segment of 4.*/
#define SHOWFIVE	0xAD		/*seven segment of 5.*/
#define SHOWSIX		0xED		/*seven segment of 6.*/
#define SHOWSEVEN	0x86		/*seven segment of 7.*/
#define SHOWEIGHT	0xEF		/*seven segment of 8.*/
#define SHOWNINE	0xAE		/*seven segment of 9.*/
#define SHOWA		0xEE		/*seven segment of A.*/
#define SHOWB		0x6D		/*seven segment of B.*/
#define SHOWC		0xE1		/*seven segment of C.*/
#define SHOWD		0x4F		/*seven segment of D.*/
#define SHOWE		0xE9		/*seven segment of E.*/
#define SHOWF		0xE8		/*seven segment of F.*/
#define ZERO		0			/*constant of 0*/
#define ONE			1			/*constant of 1*/
#define TWO			2			/*constant of 2*/
#define FOUR		4			/*constant of 4*/
#define FIVE		5			/*constant of 5*/
#define SIX			6			/*constant of 6*/
#define EIGHT		8			/*constant of 8*/
#define TWELVE		12			/*constant of 12*/
#define SIXTEEN		16			/*constant of 16*/
#define SEVENTEEN		17			/*constant of 17*/
#define EIGHTEEN		18			/*constant of 18*/
#define NINETEEN		19			/*constant of 19*/
#define TWENTYFOUR		24			/*constant of 24*/
#define TWENTYFIVE		25			/*constant of 25*/
#define TWENTYSIX		26			/*constant of 26*/
#define TWENTYSEVEN		27			/*constant of 27*/
#define SINGLEBITMASK			0x1			/*constant of 0x1*/
#define FOURBITMASK				0xF			/*constant of 0xF*/
#define OUTTWOBITMASK			0x9			/*constant of 0x9*/
#define INTWOBITMASK			0xF0		/*constant of 0xF*/

		
uint8_t value;					/* global variable that stores the button value.*/
uint8_t store;					/* global variable that stores the button value.*/
int flag;						/* global variable that prevents the spam happens.*/
unsigned char command_button;	/* value store opcode MTCP_BIOC_ON.*/
unsigned char command_led;		/* value store opcode MTCP_LED_USR.*/
unsigned char command_set;		/* value store opcode MTCP_LED_SET.*/
unsigned long flags;			/* flags saved for spinlock.*/
unsigned char tempdown;			/* variablr stores the new down button value.*/
unsigned char templeft;			/* variablr stores the new left button value.*/
unsigned char argarray[FIVE];		/* array stores the bitmask and 4 possible led values.*/
unsigned char offset;			/* variable store how much elements in argarray need to bt put.*/
unsigned char b16;				/* bit for judging whether led0 is on.*/
unsigned char b17;				/* bit for judging whether led1 is on.*/
unsigned char b18;				/* bit for judging whether led2 is on.*/
unsigned char b19;				/* bit for judging whether led3 is on.*/
int counter;					/* initialize counter.*/
static spinlock_t BUTTONLOCK = SPIN_LOCK_UNLOCKED;	/* initialize spinlock for button.*/
static spinlock_t LEDLOCK = SPIN_LOCK_UNLOCKED;		/* initialize spinlock for LED.*/

/* initializes an array that contains all the value of 16 7-segments values.*/
unsigned char segmentarr[SIXTEEN] = {SHOWZERO, SHOWONE, SHOWTWO, SHOWTHREE, SHOWFOUR, SHOWFIVE, SHOWSIX, SHOWSEVEN, SHOWEIGHT, SHOWNINE, SHOWA, SHOWB, SHOWC, SHOWD, SHOWE, SHOWF};
/************************ Protocol Implementation *************************/

/* tuxctl_handle_packet()
 * IMPORTANT : Read the header for tuxctl_ldisc_data_callback() in 
 * tuxctl-ld.c. It calls this function, so all warnings there apply 
 * here as well.
 */
 /*
 * tuxctl_handle_packet
 *   DESCRIPTION: this function contains the information caused by the tux interruption we need.
 *   INPUTS: tty_struct* tty: a tty struct for communicating information.
 *			 unsigned char* packet: a three byte array pointer which contains different return values.
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: change the value of several global variables such as button value and flag.
 */
void tuxctl_handle_packet (struct tty_struct* tty, unsigned char* packet)
{
    unsigned a, b, c;
	
    a = packet[0]; /* Avoid printk() sign extending the 8-bit */
    b = packet[1]; /* values when printing them. */
    c = packet[2];
	
	switch(a){
		
		/* if packet[0] == ACK, then it shows that LED is set completely.*/
		case MTCP_ACK:
			/* reset the flag to enable new LED writing.*/
			flag = ZERO;
			return;
		
		/* if packet[0] == MTCP_BIOC_EVENT, then it shows that a button is pressed.*/
		case MTCP_BIOC_EVENT:
			/* lock the button lock to avoid race condition with ioctl function.*/
			spin_lock_irqsave(&BUTTONLOCK, flags);
			tempdown = (c >> TWO) & SINGLEBITMASK;
			templeft = (c >> ONE) & SINGLEBITMASK;
			c = c & OUTTWOBITMASK;
			tempdown = tempdown << FIVE;
			templeft = templeft << SIX;
			
			/* renew the 8-bit button value and store it to global .*/
			store = ((c << FOUR) & INTWOBITMASK) | (b & FOURBITMASK) | tempdown | templeft;
			
			/* unlock the lock to enable read button value.*/
			spin_unlock_irqrestore(&BUTTONLOCK, flags);
			return;
		
		case MTCP_RESET:
		
			/* lock the button lock to avoid race condition with ioctl function.*/
			spin_lock_irqsave(&LEDLOCK, flags);
			
			/* enable interrupt on tux controller.*/
			command_button = MTCP_BIOC_ON;
			tuxctl_ldisc_put(tty, &command_button, ONE);
			
			/* set tux controller to led user mode.*/
			command_led = MTCP_LED_USR;
			tuxctl_ldisc_put(tty, &command_led, ONE);
			
			/* set tux controller to led set mode in user mode.*/
			command_set = MTCP_LED_SET;
			tuxctl_ldisc_put(tty, &command_set, ONE);
			
			/* put the stored LED value to show them on the tux LED.*/
			tuxctl_ldisc_put(tty, argarray, offset);
			
			/* unlock the lock to enable show LED value.*/
			spin_unlock_irqrestore(&LEDLOCK, flags);
			return;
	}
	
	
    /*printk("packet : %x %x %x\n", a, b, c); */
}

/******** IMPORTANT NOTE: READ THIS BEFORE IMPLEMENTING THE IOCTLS ************
 *                                                                            *
 * The ioctls should not spend any time waiting for responses to the commands *
 * they send to the controller. The data is sent over the serial line at      *
 * 9600 BAUD. At this rate, a byte takes approximately 1 millisecond to       *
 * transmit; this means that there will be about 9 milliseconds between       *
 * the time you request that the low-level serial driver send the             *
 * 6-byte SET_LEDS packet and the time the 3-byte ACK packet finishes         *
 * arriving. This is far too long a time for a system call to take. The       *
 * ioctls should return immediately with success if their parameters are      *
 * valid.                                                                     *
 *                                                                            *
 ******************************************************************************/
 /*
 * tuxctl_ioctl
 *   DESCRIPTION: this function contains the information caused by the tux interruption we need.
 *   INPUTS: tty_struct* tty: a tty struct for communicating information.
 *			 struct file* file: a file struct pointer.
 *			 unsigned cmd: an unsigned int number which contains different command for different cases in the ioctl function.
 *			 unsigned long arg: contains pointer or long integers that used for return values or user space pointer.
 *   OUTPUTS: none
 *   RETURN VALUE: 0 or -EINVAL
 *   SIDE EFFECTS: change the value of several global variables such as button value and flag.
 */
int tuxctl_ioctl (struct tty_struct* tty, struct file* file, 
	      unsigned cmd, unsigned long arg)
{
    switch (cmd) {
	case TUX_INIT:
	
		/* enable interrupt on tux controller.*/
		command_button = MTCP_BIOC_ON;
		tuxctl_ldisc_put(tty, &command_button, ONE);
		
		/* set tux controller to led user mode.*/
		command_led = MTCP_LED_USR;
		tuxctl_ldisc_put(tty, &command_led, ONE);
		
		/* initialize the LED array value.*/
		counter = ONE;
		while (counter < SIX)
		{
			argarray[counter] = ZERO;
			counter++;
		}
		argarray[ZERO] = FOURBITMASK;
		
		/* initialize flag and button value.*/
		flag = ZERO;
		store = ZERO;
		return 0;
	case TUX_BUTTONS:
		if (arg == ZERO)
			return -EINVAL;
		
		/* lock the button lock to avoid race condition with handle_packet function.*/
		spin_lock_irqsave(&BUTTONLOCK, flags);
		value = store;
		
		/* use copy to user function to copy the kernel button value to user mode variable value.*/
		copy_to_user((unsigned int *)arg, &value, ONE);
		
		/* unlock the lock to enable button interrupt.*/
		spin_unlock_irqrestore(&BUTTONLOCK, flags);
		return 0;
	case TUX_SET_LED:
	
		/* if a LED value is showed in process, driver could not accept another value.*/
		if (flag == ONE)
			return 0;
		
		/* set the flag value to avoid race condition.*/
		flag = ONE;
		spin_lock_irqsave(&LEDLOCK, flags);
		
		/* get four bitmask bits for 4 LEDs.*/
		b16 = (arg >> SIXTEEN) & SINGLEBITMASK;
		b17 = (arg >> SEVENTEEN) & SINGLEBITMASK;
		b18 = (arg >> EIGHTEEN) & SINGLEBITMASK;
		b19 = (arg >> NINETEEN) & SINGLEBITMASK;
		
		/* calculate new bitmask value in the first element of LED array.*/
		argarray[ZERO] = (unsigned char)(b16 + b17 * TWO + b18 * FOUR + b19 * EIGHT);
		offset = ONE;	
		
		/* if first LED should be on, we need its value.*/
		if (b16 == ONE)
		{
			/* find the correct 7-segment value in segmentarr array.*/
			argarray[offset] = segmentarr[(arg & FOURBITMASK)];
			
			/* determine whether the decimal point needs to be on.*/
			b16 = (arg >> TWENTYFOUR) & SINGLEBITMASK;
			if (b16 == ONE)
				argarray[offset] += SIXTEEN;
			
			/* increment the index of LED array.*/
			offset++;
		}
		/* if second LED should be on, we need its value.*/
		if (b17 == ONE)
		{
			/* find the correct 7-segment value in segmentarr array.*/
			argarray[offset] = segmentarr[((arg >> FOUR) & FOURBITMASK)];
			
			/* determine whether the decimal point needs to be on.*/
			b17 = (arg >> TWENTYFIVE) & SINGLEBITMASK;
			if (b17 == ONE)
				argarray[offset] += SIXTEEN;
			
			/* increment the index of LED array.*/
			offset++;
		}
		/* if third LED should be on, we need its value.*/
		if (b18 == ONE)
		{
			/* find the correct 7-segment value in segmentarr array.*/
			argarray[offset] = segmentarr[((arg >> EIGHT) & FOURBITMASK)];
			
			/* determine whether the decimal point needs to be on.*/
			b18 = (arg >> TWENTYSIX) & SINGLEBITMASK;
			if (b18 == ONE)
				argarray[offset] += SIXTEEN;
			
			/* increment the index of LED array.*/
			offset++;
		}
		/* if third LED should be on, we need its value.*/
		if (b19 == ONE)
		{
			/* find the correct 7-segment value in segmentarr array.*/
			argarray[offset] = segmentarr[((arg >> TWELVE) & FOURBITMASK)];
			
			/* determine whether the decimal point needs to be on.*/
			b19 = (arg >> TWENTYSEVEN) & SINGLEBITMASK;
			if (b19 == ONE)
				argarray[offset] += SIXTEEN;
			
			/* increment the index of LED array.*/
			offset++;
		}
		
		/* set tux controller to led set mode in user mode.*/
		command_set = MTCP_LED_SET;
		tuxctl_ldisc_put(tty, &command_set, ONE);
		
		/* put the stored LED value to show them on the tux LED.*/
		tuxctl_ldisc_put(tty, argarray, offset);
		
		/* unlock the lock to enable reset LED value.*/
		spin_unlock_irqrestore(&LEDLOCK, flags);
		return 0;	
	case TUX_LED_ACK:
	case TUX_LED_REQUEST:
	case TUX_READ_LED:
	default:
	    return -EINVAL;
    }
}

