
/*
	Reference: wiki.osdev.org Paging section
*/
#include "page.h"

#define PHYS_ADDR_VIDEOMEM 0x000B8000

#define ONE_KB	 	1024
#define FOUR_KB  	4096
#define ONE			1
#define TWO			2
#define THREE		3
#define FOUR		4
#define FIVE		5
#define SIX			6
#define THIRTYFOUR	34

#define FOUR_MB	 0x00400000
#define SIZE_MB_ON	0x80

#define RW_P_ENABLE	0x00000003
#define User 0x00000004

/*
 * enable_paging
 *   DESCRIPTION: enables paging by loading CR3 with address of pde and set the PG PE bits of CR0
 *   INPUTS: address of the paging directory
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: CR0 and CR3 changed
 */
void enable_paging(int * pde_enable) {
	__asm__ volatile(
		"movl 8(%%ebp), %%eax\n"
		"movl %%eax, %%cr3\n"
		
		"movl %%cr0, %%eax\n"
		"orl  $0x80000000, %%eax\n"
		"movl %%eax, %%cr0\n"
		:
		:
		:"%eax"
	); 
}
/*
 * enable_PSE
 *   DESCRIPTION: enable the PSE bit for using 4MB pages
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: cr4(PSE) modified
 */
void enable_PSE() {
	__asm__ volatile(
		"movl %%cr4, %%eax\n"
		"orl  $0x00000010, %%eax\n"
		"movl %%eax, %%cr4\n"
		:
		:
		:"%eax"
	);
}
/*
 * map_page_init
 *   DESCRIPTION: initialze the paging, initialize the pte array for first pde index, kernel for second pde index and zero for the rest of pde
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: fill in the pde and pte array, maps the virtual memory to physical memory
 */
void map_page_init(){
	/*pde and pte flags for setting lower 12 bits of each entry in the directory and table*/
	unsigned long pde_flag;
	unsigned long pte_flag;
	//int pde[ONE_KB] __attribute__ ((aligned (FOUR_KB)));
	//int pte_zero[ONE_KB] __attribute__ ((aligned (FOUR_KB)));
	/*counter*/
	int i;
	/*set the present bit high and enable R/W functionality which are lowest two bits*/
	pde_flag = RW_P_ENABLE;
	pte_flag = RW_P_ENABLE;
	pde_arr[0] = pde_0;
	pde_arr[ONE] = pde_1;
	pde_arr[TWO] = pde_2;
	pde_arr[THREE] = pde_3;
	pde_arr[FOUR] = pde_4;
	pde_arr[FIVE] = pde_5;
	pde_arr[SIX] = pde_6;
	/* manually set first entry of the directory as 4-kb aligned address of page table we established and set the flag*/
	pde_arr[0][0] = (uint32_t)pte_zero| pde_flag;
	/*set the Physical address of video memory into its corresponding index in page table*/
	pte_zero[PHYS_ADDR_VIDEOMEM/FOUR_KB] =  (PHYS_ADDR_VIDEOMEM)| pte_flag;
	pte_zero[PHYS_ADDR_VIDEOMEM/FOUR_KB+ONE] =  (PHYS_ADDR_VIDEOMEM+FOUR_KB)| pte_flag;
	pte_zero[PHYS_ADDR_VIDEOMEM/FOUR_KB+TWO] =  (PHYS_ADDR_VIDEOMEM+FOUR_KB*TWO)| pte_flag;
	pte_zero[PHYS_ADDR_VIDEOMEM/FOUR_KB+THREE] =  (PHYS_ADDR_VIDEOMEM+FOUR_KB*THREE)| pte_flag;
	
	/* update the count bit in first entry indicating starting address is 4MB(starting from the 22th bit)
	 * raise the Page size bit high(which is the 7th bit) and set the flag*/
	pde_arr[0][1] = FOUR_MB|SIZE_MB_ON|pde_flag;
	//pde[1] = (1<<22)|(1<<7)|pde_flag;
	/* loop through the page table and disable all entries other than video memory*/
	for(i = 0; i < ONE_KB; i++){
		//if ((i != (PHYS_ADDR_VIDEOMEM/FOUR_KB))&&(i != (PHYS_ADDR_VIDEOMEM/FOUR_KB+1))&&(i != (PHYS_ADDR_VIDEOMEM/FOUR_KB+TWO))&&(i != (PHYS_ADDR_VIDEOMEM/FOUR_KB+THREE))) {
		//	pte_zero[i]=0;
		pte_zero[i]=i*FOUR_KB|pte_flag;
		
	}
	
	/* enable PSE*/
	enable_PSE();
	/* enable paging by loading CR3 with pde address*/
	enable_paging(pde_arr[0]);
	
	return;
}
/*
 * set_page
 *	 DESCRIPTION: set page for each program called in the syatem call.
 *   INPUTS: pde_idx: the index of page directory table.
 *				  phys_addr: physical address that we want to page with the virtual memory.
 *				  index: index of the index array.
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: fill in the pde and pte array, maps the virtual memory to physical memory
 */
void set_page(uint32_t pde_idx, uint32_t phys_addr,int index, int fourkb_on,int vid_idx) {
	
	/*pde and pte flags for setting lower 12 bits of each entry in the directory and table*/
	unsigned long pde_flag;
	unsigned long pte_flag;
	/*counter*/
	int i;
	int phys_videomem = (vid_idx+1)*FOUR_KB+PHYS_ADDR_VIDEOMEM;
	/*set the present bit high and enable R/W functionality which are lowest two bits*/
	pde_flag = RW_P_ENABLE;
	pte_flag = RW_P_ENABLE;
	
	
	/* manually set first entry of the directory as 4-kb aligned address of page table we established and set the flag*/
	pde_arr[index][0] = (uint32_t)pte_zero| pde_flag;
	/*set the Physical address of video memory into its corresponding index in page table*/
	pte_zero[PHYS_ADDR_VIDEOMEM/FOUR_KB] =  PHYS_ADDR_VIDEOMEM| pte_flag;
	pte_zero[PHYS_ADDR_VIDEOMEM/FOUR_KB+ONE] =  (PHYS_ADDR_VIDEOMEM+FOUR_KB)| pte_flag;
	pte_zero[PHYS_ADDR_VIDEOMEM/FOUR_KB+TWO] =  (PHYS_ADDR_VIDEOMEM+FOUR_KB*TWO)| pte_flag;
	pte_zero[PHYS_ADDR_VIDEOMEM/FOUR_KB+THREE] =  (PHYS_ADDR_VIDEOMEM+FOUR_KB*THREE)| pte_flag;
	
	/* update the count bit in first entry indicating starting address is 4MB(starting from the 22th bit)
	 * raise the Page size bit high(which is the 7th bit) and set the flag*/
	pde_arr[index][ONE] = FOUR_MB|SIZE_MB_ON|pde_flag;
	/* loop through the page table and disable all entries other than video memory*/
	for(i = 0; i < ONE_KB; i++){
		//if ((i != ((PHYS_ADDR_VIDEOMEM)/FOUR_KB))&&(i != ((PHYS_ADDR_VIDEOMEM)/FOUR_KB+ONE))&&(i != ((PHYS_ADDR_VIDEOMEM)/FOUR_KB+TWO))&&(i != ((PHYS_ADDR_VIDEOMEM)/FOUR_KB+THREE))) {
		//	pte_zero[i]=0;
		
		pte_zero[i]=i*FOUR_KB|pte_flag;
		
	}

	pde_arr[index][pde_idx] = phys_addr|SIZE_MB_ON|RW_P_ENABLE|User;
	/*if use the vid_map, then set up another virtual memory*/
	if (fourkb_on) {
		pde_arr[index][THIRTYFOUR] = (uint32_t)pte_one| pde_flag | User;
		pte_one[1] = (phys_videomem)| pte_flag | User;
	}
	/* enable PSE*/
	enable_PSE();
	enable_paging(pde_arr[index]);
	/* enable paging by loading CR3 with pde address*/
	return;
}


/*
 * set_page_1
 *	 DESCRIPTION: set page for each program called in the syatem call.
 *   INPUTS: pde_idx: the index of page directory table.
 *				  phys_addr: physical address that we want to page with the virtual memory.
 *				  index: index of the index array.
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: fill in the pde and pte array, maps the virtual memory to physical memory
 */
void set_page_1(uint32_t pde_idx, uint32_t phys_addr,int index, int fourkb_on,int vid_idx,int curr_terminal) {
	
	/*pde and pte flags for setting lower 12 bits of each entry in the directory and table*/
	unsigned long pde_flag;
	unsigned long pte_flag;
	/*counter*/
	int i;
	int phys_videomem = (curr_terminal+1)*FOUR_KB+PHYS_ADDR_VIDEOMEM;
	/*set the present bit high and enable R/W functionality which are lowest two bits*/
	pde_flag = RW_P_ENABLE;
	pte_flag = RW_P_ENABLE;
	
	
	/* manually set first entry of the directory as 4-kb aligned address of page table we established and set the flag*/
	pde_arr[index][0] = (uint32_t)pte_zero| pde_flag;
	/*set the Physical address of video memory into its corresponding index in page table*/
	pte_zero[PHYS_ADDR_VIDEOMEM/FOUR_KB] =  PHYS_ADDR_VIDEOMEM| pte_flag;
	pte_zero[PHYS_ADDR_VIDEOMEM/FOUR_KB+ONE] =  (PHYS_ADDR_VIDEOMEM+FOUR_KB)| pte_flag;
	pte_zero[PHYS_ADDR_VIDEOMEM/FOUR_KB+TWO] =  (PHYS_ADDR_VIDEOMEM+FOUR_KB*TWO)| pte_flag;
	pte_zero[PHYS_ADDR_VIDEOMEM/FOUR_KB+THREE] =  (PHYS_ADDR_VIDEOMEM+FOUR_KB*THREE)| pte_flag;
	
	/* update the count bit in first entry indicating starting address is 4MB(starting from the 22th bit)
	 * raise the Page size bit high(which is the 7th bit) and set the flag*/
	pde_arr[index][ONE] = FOUR_MB|SIZE_MB_ON|pde_flag;
	/* loop through the page table and disable all entries other than video memory*/
	for(i = 0; i < ONE_KB; i++){
		//if ((i != ((PHYS_ADDR_VIDEOMEM)/FOUR_KB))&&(i != ((PHYS_ADDR_VIDEOMEM)/FOUR_KB+ONE))&&(i != ((PHYS_ADDR_VIDEOMEM)/FOUR_KB+TWO))&&(i != ((PHYS_ADDR_VIDEOMEM)/FOUR_KB+THREE))) {
		//	pte_zero[i]=0;
		pte_zero[i]=i*FOUR_KB|pte_flag;
		
	}
	
	pde_arr[index][pde_idx] = phys_addr|SIZE_MB_ON|RW_P_ENABLE|User;
	
	if (fourkb_on) {
		pde_arr[index][THIRTYFOUR] = (uint32_t)pte_one| pde_flag | User;
		pte_one[curr_terminal+1] = (phys_videomem)| pte_flag | User;
	}
	/* enable PSE*/
	enable_PSE();
	enable_paging(pde_arr[index]);
	/* enable paging by loading CR3 with pde address*/
	return;
}
