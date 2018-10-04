#ifndef PAGE_H
#define PAGE_H

#include "types.h"
#include "lib.h"
/*
	Reference: wiki.osdev.org Paging section
*/

/*enable paging*/
void enable_paging(int * pde_enable);

/*enable PSE for 4MiB pages*/
void enable_PSE();

/*initialize paging*/
void map_page_init();

/* set page function that used in the system call.*/
void set_page(uint32_t pde_idx, uint32_t phys_addr,int index, int fourkb_on,int vid_idx);

/* set page function that used in the system call with vid_map virtual memory available.*/
void set_page_1(uint32_t pde_idx, uint32_t phys_addr,int index, int fourkb_on,int vid_idx, int curr_terminal);

#endif

