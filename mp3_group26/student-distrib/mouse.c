
#include "mouse.h"

int mouse_x;
int mouse_y;
/*int mouse_cycle = 0;
int8_t mouse_byte[3];

void mouse_wait(uint8_t a_type) //unsigned char
{
  int _time_out=100000; //unsigned int
  if(a_type==0)
  {
    while(_time_out--) //Data
    {
      if((inb(0x64) & 1)==1)
      {
        return;
      }
    }
    return;
  }
  else
  {
    while(_time_out--) //Signal
    {
      if((inb(0x64) & 2)==0)
      {
        return;
      }
    }
    return;
  }
}

void mouse_write(uint8_t a_write) //unsigned char
{
  //Wait to be able to send a command
  mouse_wait(1);
  //Tell the mouse we are sending a command
  outb(0xD4, 0x64);
  //Wait for the final part
  mouse_wait(1);
  //Finally write
  outb(a_write, 0x60);
}

uint8_t mouse_read()
{
  //Get's response from mouse
  mouse_wait(0); 
  return inb(0x60);
}*/


void mouse_init() {	
	cli();
	int i;
	uint8_t temp;
	mouse_x = 40;
	mouse_y = 13;
	*(uint8_t *)(0xB8000+mouse_y*160+mouse_x*2) = '?';
	enable_irq(12);
	
  //Enable the auxiliary mouse device
	//mouse_wait(1);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(0xA8, 0x64);
  
  //Enable the interrupts
	//mouse_wait(1);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(0x20, 0x64);
	//mouse_wait(0);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 1)==1) break;
	}
	temp=((inb(0x60)&0xDF) | 2);
	//mouse_wait(1);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(0x60, 0x64);
	//mouse_wait(1);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(temp, 0x60);
 
 //set MouseID
	//mouse_write(0xF3);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(0xD4, 0x64);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(0xF3, 0x60);
	//mouse_write((uint8_t)200);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(0xD4, 0x64);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(200, 0x60);
	//mouse_write(0xF3);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(0xD4, 0x64);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(0xF3, 0x60);
	//mouse_write((uint8_t)100);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(0xD4, 0x64);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(100, 0x60);
	//mouse_write(0xF3);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(0xD4, 0x64);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(0xF3, 0x60);
	//mouse_write((uint8_t)80);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(0xD4, 0x64);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(80, 0x60);
	
  //Tell the mouse to use default settings
	//mouse_write(0xF6);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(0xD4, 0x64);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(0xF6, 0x60);
	//mouse_read();  //Acknowledge
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 1)==1) break;
	}
    inb(0x60);
  
  //Enable the mouse
	//mouse_write(0xF4);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(0xD4, 0x64);
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 2)==0) break;
	}
	outb(0xF4, 0x60);
	//mouse_read();  //Acknowledge
	for (i=0; i<100000; i++) {
		if((inb(0x64) & 1)==1) break;
	}
    inb(0x60);
	sti();
}

void mouse_interrupt_handler() {
	cli();
	
	char temp_buf[128];
	int i;
	uint8_t mouse_flag;
	uint8_t packet[4];  //check!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	int delta_mouse_x = 0;
	int delta_mouse_y = 0;
	
	for (i=0; i<128; i++) {
		temp_buf[i] = '\0';
	}
	
	mouse_flag = inb(0x64);
	if (((mouse_flag)&0x21) != 0x21) {
		send_eoi(12);
		sti();
		asm("leave");
		asm("iret");
	}
	packet[0] = inb(0x60);
	if ((packet[0]&0xC0)!= 0) {
		send_eoi(12);
		sti();
		asm("leave");
		asm("iret");
	}
	
	packet[1] = inb(0x60);
	
	packet[2] = inb(0x60);
	
	packet[3] = inb(0x60);
	
	if ((packet[0]&0x10)!=0) delta_mouse_x = (int)(0xFFFFFF00|packet[1]);
	else delta_mouse_x = (int)packet[1];
	if ((packet[0]&0x20)!=0) delta_mouse_y = (int)(0xFFFFFF00|packet[2]);
	else delta_mouse_y = (int)packet[2];
	
	*(uint8_t *)(0xB8000+mouse_y*160+mouse_x*2) = *(uint8_t *)(terminals[curr_terminal_idx].video_mem+mouse_y*160+mouse_x*2);
	*(uint8_t *)(0xB8000+mouse_y*160+mouse_x*2+1) = *(uint8_t *)(terminals[curr_terminal_idx].video_mem+mouse_y*160+mouse_x*2+1);
	
	mouse_x = mouse_x + (int)(delta_mouse_x/5);
	mouse_y = mouse_y - (int)(delta_mouse_y/10);
	
	if (mouse_x>79) {
		mouse_x = 78;
	}
	else if (mouse_x<0) {
		mouse_x = 0;
	}
	if (mouse_y>24) {
		mouse_y = 24;
	}
	else if (mouse_y<0) {
		mouse_y = 0;
	}
	*(uint8_t *)(0xB8000+mouse_y*160+mouse_x*2) = '?';
	*(uint8_t *)(0xB8000+mouse_y*160+mouse_x*2+1) = 10;
	/* if left click */
	if ((packet[0]&0x01) == 1) {
		i = 0;
		while (((*(uint8_t *)(terminals[curr_terminal_idx].video_mem+mouse_y*160+i*2))!='\0')&&((*(uint8_t *)(terminals[curr_terminal_idx].video_mem+mouse_y*160+i*2))!=' ')) {
			temp_buf[i] = *(char *)(terminals[curr_terminal_idx].video_mem+mouse_y*160+i*2);
			i++;
		}
		temp_buf[i] = '\0';
		for (i=0; i<128; i++) {
			terminals[curr_terminal_idx].typed_buf[i] = '\0';
			terminals[curr_terminal_idx].curr_char_idx = 0;
		}
		i=0;
		while (temp_buf[i] != '\0') {
			putc(temp_buf[i]);
			terminals[curr_terminal_idx].typed_buf[i]=temp_buf[i];
			terminals[curr_terminal_idx].curr_char_idx++;
			i++;
		}
	} 
	
	send_eoi(12);
	sti();
	asm("leave");
	asm("iret");
}
