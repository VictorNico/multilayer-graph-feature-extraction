#ifndef HELLO_WORDL_MAP_REDUCE_H
#define HELLO_WORDL_MAP_REDUCE_H

#define NB_THREADS 5
#define TAILLE 1500

#include <pthread.h>
#include <string.h>

typedef struct
{
	float *x;
	float *y;
	float sum;
	int length;
} data_t;

data_t data;
pthread_mutex_t mutex_variable;

char** tab_lines;
int nb_lines;

char** tab_words;
int nb_words;

char** tab_d_words;
int nb_d_words;

int* tab_occ_words; //contains the total count of each word

void *map(void *arg);

void *reduce(void *arg);


int str_split(char* str, char* car, char*reslt[]);
char ** read_file(char * file_name, int nb_line);
void print_tab_string(char** tab_line, int nb_line);
int count_lines(char * file_name);
void removeChar(char *str, char garbage);
void quicksort_string(char** tab_string, int first, int last);
void extract_distinct_words();
int count_word_occurence(char*word, char**tab_words, int nb_words);
void print_word_occurrence(char** tab_d_words, int nb_line, int* tab_occur);
void triFusion(int i, int j,char** tab_string, char**  tmp);
void string_insertion_sort(char** tab_string, int SIZE);
void write_csv(const char* line, const char* path);
int fileExists(const char* filePath) ;

#endif





