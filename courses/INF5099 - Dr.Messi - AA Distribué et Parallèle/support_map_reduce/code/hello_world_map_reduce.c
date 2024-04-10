#include "hello_world_map_reduce.h"

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/*** function used to split a line of string
 * Params:
 * 		str - vector of characters - input string line
 * 		car - character - the separator partern
 * 		reslt - vector of string - sub string resultat of split base on car character pattern
 * Return:
 * 		j_retourn - integer - the number of sub string found after split
 * 
 */
int str_split(char* str, char* car, char*reslt[])
{
	char *str1, *token;
        char *saveptr1;
        int j, j_retour;

	for (j = 1, str1 = str; ; j++, str1 = NULL) {
		// strtok_r  function in C is used to tokenize a string into smaller tokens, based on a set of delimiter characters.
		// char *strtok_r(char *str, const char *delim, char **saveptr);
       token = strtok_r(str1, car, &saveptr1);
       if (token == NULL) // if not token extract with the delimiter break and end the loop
           break;
		reslt[j-1] = token; // else save the current substring before the delimiter
		j_retour = j; // the number of substring found
    }

	return j_retour;
}

void *map(void *arg)
{
	
	int i, vector_length, j; 
	long begin, end;
	long thread_id;
	thread_id = (long)arg;
	char** tab_local = (char**)malloc(sizeof(char*));

	vector_length = nb_lines; //length of the vector of sentences

	// block paralisation logic to bound the thread's load
	begin = thread_id*(vector_length/NB_THREADS);

	if(thread_id == (NB_THREADS - 1))
	   end = nb_lines;
	else
	   end =  begin + (vector_length/NB_THREADS);

	char* resl_split[200];	
	int split_resl_length = 0, length_tab_local = 0, prev;

	// fetch on lines in the file
	// for each line split within " " pattern
	// reajust the table of substring found to integrate the new substring starting on the previous end
	// and then add the news substring to the vector of substrings
	for (i= begin; i<end ; i++)
	{
		split_resl_length = str_split(tab_lines[i]," ",resl_split);
		//printf("split_length = %d\n", split_resl_length);
		prev = length_tab_local;
		length_tab_local = length_tab_local + split_resl_length;
		tab_local = realloc(tab_local, length_tab_local*sizeof(char*));
		for(j=0;j<split_resl_length;j++)
		{
			//printf("Thread %d prev=%d j = %d prev+j=%d\n", thread_id, prev, j, prev+j);
			tab_local[prev+j] = malloc(sizeof(char)*strlen(resl_split[j]));
			strcpy(tab_local[prev+j], resl_split[j]);
		}
			
	}
	//printf("Thread %ld debut = %ld fin = %ld\n", indice, debut, fin);

	// save the local vector in the global one 
	// for that lock variable access before to limit colision
	pthread_mutex_lock (&mutex_variable);
		prev = nb_words;
		nb_words = nb_words + length_tab_local;
		tab_words = realloc(tab_words, nb_words*sizeof(char*));
		for(i=0; i<length_tab_local; i++)
		{
			tab_words[prev+i] = malloc(sizeof(char)*strlen(tab_local[i]));
			strcpy(tab_words[prev+i], tab_local[i]);
		}	
	pthread_mutex_unlock (&mutex_variable);

	pthread_exit((void*) 0);
}

void *reduce(void *arg)
{
	int i, vector_length; 
	long begin, end;
	long thread_id;
	thread_id = (long)arg;

	vector_length = nb_d_words;//length of the vector of distinct word

	begin = thread_id*(vector_length/NB_THREADS);

	if(thread_id == (NB_THREADS - 1))
	   end = nb_d_words;
	else
	   end =  begin + (vector_length/NB_THREADS);

	// count the number of occurence of subword in the vector of substring
	for (i= begin; i<end; i++)
	{
		tab_occ_words[i] = count_word_occurence(tab_d_words[i], tab_words, nb_words);
	}

	pthread_exit((void*) 0);
}

/** count occurence of a word
 * Params:
 * 		word - vector of characters - the word to count
 * 		tab_words - vector of string - vector of all words
 * 		nb_words - integer - the number of words in the tab_words
 * Return:
 * 		result - integer - the number of occurence
 * 
 */
int count_word_occurence(char*word, char**tab_words, int nb_words)
{
	int i = 0, result = 0;
	
	while(i<nb_words)
	{
		if(strcmp(word, tab_words[i])==0)
			result = result+1;
		i = i+1;
	}

	return result;
}

// open a file and read lines content's and count
int count_lines(char * file_name){
  FILE * fichier = fopen(file_name, "r");
  char ligne [TAILLE];
  int nLignes = 0;

  while(fgets(ligne,TAILLE,fichier)!=NULL)
  {
    nLignes++;
  }
    fclose(fichier);
  return nLignes;
}

void removeChar(char *str, char garbage) {
    char *src, *dst;
    for (src = dst = str; *src != '\0'; src++) {
        *dst = *src;
        if (*dst != garbage) dst++;
    }
    *dst = '\0';
}

char ** read_file(char * file_name, int n){
	char ** result;
	FILE *f= fopen(file_name,"r");
	char ligne[TAILLE];
	result = (char**)malloc(n*sizeof(char*));

	int i=0;
	while(fgets(ligne,TAILLE,f)!=NULL){
		result[i]=malloc(TAILLE);
		removeChar(ligne, '\n');
		removeChar(ligne, '.');
		removeChar(ligne, ';');
		removeChar(ligne, ',');
		removeChar(ligne, '?');
		removeChar(ligne, '!');
		strcpy(result[i],ligne);
		i++;
	}
	fclose(f);
	return result;
}


void print_word_occurrence(char** tab_d_words, int nb_line, int* tab_occur)
{
	int i=0;
	
	while(i<nb_line)
	{
		printf(" %s\t\t:%d\n", tab_d_words[i], tab_occur[i]);
		i++;
	}

}

void print_tab_string(char** tab_line, int nb_line)
{
	int i=0;
	
	while(i<nb_line)
	{
		printf("%d: %s\n", i, tab_line[i]);
		i++;
	}

}


void quicksort_string(char** tab_string, int first, int last){
   int i, j, pivot;
   char* temp;

   if(first<last){
      pivot=first;
      i=first;
      j=last;

      while(i<j){
         while((strcmp(tab_string[i], tab_string[pivot])<=0) && (i<last))
            i++;
         while(strcmp(tab_string[j], tab_string[pivot])>0)
            j--;
         if(i<j){
	    temp = (char*)malloc(sizeof(char)*strlen(tab_string[i]));
               strcpy(temp, tab_string[i]);
               strcpy(tab_string[i], tab_string[j]);
               strcpy(tab_string[j], temp);
	    free(temp);
         }
      }

    temp = (char*)malloc(sizeof(char)*strlen(tab_string[pivot]));
        strcpy(temp, tab_string[pivot]);
        strcpy(tab_string[pivot], tab_string[j]);
        strcpy(tab_string[j], temp);
    free(temp);

      quicksort_string(tab_string, first,j-1);
      quicksort_string(tab_string, j+1,last);

   }
}

void extract_distinct_words()
{
	//tab_words nb_words;
	
	int i = 0, k = 0;
	nb_d_words = k+1;
	tab_d_words = realloc(tab_d_words, nb_d_words*sizeof(char*));
	tab_d_words[k] = malloc(sizeof(char)*strlen(tab_words[i]));
	strcpy(tab_d_words[k], tab_words[k]);

	while(i<nb_words)
	{
		if(strcmp(tab_d_words[k],tab_words[i])!=0)
		{
			k = k+1;
			nb_d_words = k+1;
			tab_d_words = realloc(tab_d_words, nb_d_words*sizeof(char*));
			tab_d_words[k] = malloc(sizeof(char)*strlen(tab_words[i]));
			strcpy(tab_d_words[k], tab_words[i]);
		}
		i = i+1;
	}
}
