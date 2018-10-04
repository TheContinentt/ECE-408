/*
	Reference: wiki.osdev.org PC speaker section
*/
#include "sound.h"

uint32_t sound_freq = 1200;

void play_sound(uint32_t nFrequence){
	uint32_t Div;
 	uint8_t tmp;
 
 	Div = 1193180 / nFrequence;
 	outb(0xb6,0x43);
 	outb((uint8_t) (Div & 0xFF), 0x42);
 	outb((uint8_t) (Div >> 8), 0x42);
 
 	tmp = inb(0x61);
  	
	if (tmp != (tmp | 3))
 	outb( (tmp | 3), 0x61);
 	
}
 
void nosound(){
	cli();
	uint8_t tmp = inb(0x61) & 0xFC;
 	outb(tmp, 0x61);
	sti();
}

void beep() {
	if (sound_freq > 1100)
		sound_freq = 200;
	sound_freq = sound_freq + 40;
 	play_sound(sound_freq);
 }
 
 void beep_init() {
	if (sound_freq < 200)
		sound_freq = 1200;
	sound_freq = sound_freq - 100;
	play_sound(sound_freq);
}
