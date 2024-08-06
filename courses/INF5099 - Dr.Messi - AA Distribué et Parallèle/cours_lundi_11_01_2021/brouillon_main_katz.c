#include<stdio.h>
#include <limits.h>
#include"time_manager.h"
#include"thread_management.h"
#include <sys/time.h>
#include"function.h"
#include"graph_community.h"
#include"graph_base.h"
#include"graph_bloc.h"
#include"graph_yale.h"
#include"katz.h"
#include"set.h"
//#include <papi.h>
#include"index_mapper.h"

timeval_t t1, t2;
timezone_t tz;

static struct timeval _t1, _t2;
static struct timezone _tz;
static unsigned long _temps_residuel = 0;
#define top1() gettimeofday(&_t1, &_tz)
#define top2() gettimeofday(&_t2, &_tz)

void init_cpu_time(void)
{
   top1(); top2();
   _temps_residuel = 1000000L * _t2.tv_sec + _t2.tv_usec -
                     (1000000L * _t1.tv_sec + _t1.tv_usec );
}

unsigned long cpu_time(void) /* retourne des microsecondes */
{
   return 1000000L * _t2.tv_sec + _t2.tv_usec -
           (1000000L * _t1.tv_sec + _t1.tv_usec ) - _temps_residuel;
}

void gml2txt(char* file_name);
void txt2gml(char* file_name);
void yale2txt(graph_yale_t G_yale, char* file_name, char*optimization); 
void bloc2txt(graph_bloc_t G_bloc, char* file_name, char*optimization); 

int main(int argc, char* argv[])
{
	if(argc == 3)
	{
	    switch(atoi(argv[2]))
	    {
		case 1:
	     		gml2txt(argv[1]);
			return 1;
		case 2:
			txt2gml(argv[1]);
			return 1;
		default:
			printf("./main nom_fich.gml/txt 1/2\n");
			return 1;
	    }

	}
	else
	   if(argc!=7)
	   {
		printf("./main nom_fich.tree nom_fich.txt nom_fich.gml max_l nb_thread data_structure\n");
		return 1;
	   }

	char*nomFichTree = malloc(sizeof(char)*50), *nomFichTxt = malloc(sizeof(char)*50), *nomFichGml = malloc(sizeof(char)*50),nom_fich[200];
	bool oriented = true;
	float* beta = malloc(sizeof(float)); *beta = 1.0;
	long int cost1 = 0, cost2 = 0;
	int max_l =  atoi(argv[4]), nb_node, i, j, nb_thread= atoi(argv[5]),ds = atoi(argv[6]);
	void* node;
	unsigned long time_neig, time_build_path;
	node_b_i_t x,y;
	g_param_t* g_param = malloc(sizeof(g_param_t));
	h_table_t* h_table = malloc(sizeof(h_table_t));
	index_mapper_t* index_mapper;
	graph_community_t* g_comm;
	graph_base_t* g_base;
	graph_bloc_t* g_bloc;
	graph_yale_t* g_yale;
	strcpy(nomFichTree,argv[1]);strcpy(nomFichTxt, argv[2]);strcpy(nomFichGml,argv[3]);

	switch(ds)
	{
	case 3:
		top1 ();
		g_base = malloc(sizeof(graph_base_t));
		load_graph_gml(nomFichGml, g_base);
		g_bloc = get_graph_bloc(*g_base);
		bloc2txt(*g_bloc, nomFichTxt, "3_simple");

		free_g_base(g_base);
		free_g_bloc(g_bloc);
		break;
	case 4:
		top1 ();
		g_base = malloc(sizeof(graph_base_t));
		load_graph_gml(nomFichGml, g_base);
		g_yale = get_graph_yale(*g_base);
		yale2txt(*g_yale, nomFichTxt, "4_simple");

		free_g_base(g_base);
		free_g_yale(g_yale);
		break;
	case 5:
		top1 ();
		ds = 3;
		g_comm = malloc(sizeof(graph_community_t));
		*g_comm = init_graph_community(nomFichTree, nomFichTxt, oriented, h_table, g_param);//graphe taken from txt file are oriented
		index_mapper = do_index_mapping(*g_comm);
		g_bloc = comm2bloc(*g_comm, index_mapper);
		bloc2txt(*g_bloc, nomFichTxt, "5_numbering");

		delete_mapper(index_mapper);
		free_g_param(g_param);
		free_h_table(h_table);
		free(h_table);
		free_g_comm(g_comm);
		free_g_bloc(g_bloc);
		ds = 5;
		break;
	case 6:
		top1 ();
		ds = 4;
		g_comm = malloc(sizeof(graph_community_t));
		*g_comm = init_graph_community(nomFichTree, nomFichTxt, oriented, h_table, g_param);//graphe taken from txt file are oriented
		index_mapper = do_index_mapping(*g_comm);
		g_yale = comm2yale_2(*g_comm, index_mapper, *g_param);
		yale2txt(*g_yale, nomFichTxt, "6_numbering");

		delete_mapper(index_mapper);
		free_g_param(g_param);
		free_h_table(h_table);
		free(h_table);
		free_g_comm(g_comm);
	 	//print_mapped_graph_yale(*g_yale, *index_mapper);
		free_g_yale(g_yale);
		ds = 6;
		break;
	case 7:
		top1 ();
		ds = 3;
		g_comm = malloc(sizeof(graph_community_t));
		*g_comm = init_graph_community(nomFichTree, nomFichTxt, oriented, h_table, g_param);//graphe taken from txt file are oriented
		index_mapper = do_index_mapping_sorted_comm(*g_comm);
		g_bloc = comm2bloc(*g_comm, index_mapper);
		bloc2txt(*g_bloc, nomFichTxt, "7_numbering_comm");

		delete_mapper(index_mapper);
		free_g_param(g_param);
		free_h_table(h_table);
		free(h_table);
		free_g_comm(g_comm);
		free_g_bloc(g_bloc);
		ds = 7;
		break;
	case 8:
		top1 ();
		ds = 4;
		g_comm = malloc(sizeof(graph_community_t));
		*g_comm = init_graph_community(nomFichTree, nomFichTxt, oriented, h_table, g_param);//graphe taken from txt file are oriented 
		index_mapper = do_index_mapping_sorted_comm(*g_comm);
		g_yale = comm2yale_2(*g_comm, index_mapper, *g_param);
		yale2txt(*g_yale, nomFichTxt, "8_numbering_comm");

		delete_mapper(index_mapper);
		free_g_param(g_param);
		free_h_table(h_table);
		free(h_table);
		free_g_comm(g_comm);
		free_g_yale(g_yale);
		ds = 8;
		break;
	case 9:
		top1 ();
		ds = 3;
		g_comm = malloc(sizeof(graph_community_t));
		*g_comm = init_graph_community_sorted_neighbors(nomFichTree, nomFichTxt, oriented, h_table, g_param);//graphe taken from txt file are oriented, so oriented<-true
		index_mapper = do_index_mapping_sorted_comm(*g_comm);
		g_bloc = comm2bloc(*g_comm, index_mapper);
		bloc2txt(*g_bloc, nomFichTxt, "9_sorted_neigh-numbering_comm");

	 	//print_graph_bloc(*g_bloc);
		free_h_table(h_table);
		free(h_table);
		free_g_param(g_param);
		free_g_comm(g_comm);
	//	delete_mapper(index_mapper);
	 	//print_mapped_graph_bloc(*g_bloc, *index_mapper);
		delete_mapper(index_mapper);
		free_g_bloc(g_bloc);
		ds = 9;
		break;
	case 10:
		top1 ();
		ds = 4;
		g_comm = malloc(sizeof(graph_community_t));
		*g_comm = init_graph_community_sorted_neighbors(nomFichTree, nomFichTxt, oriented, h_table, g_param);//graphe taken from txt file are oriented, so oriented<-true
		index_mapper = do_index_mapping_sorted_comm(*g_comm);
		g_yale = comm2yale_2(*g_comm, index_mapper, *g_param);
		yale2txt(*g_yale, nomFichTxt, "10_sorted_neigh-numbering_comm");

	 	//print_graph_yale(*g_yale);
		free_h_table(h_table);
		free_g_param(g_param);
		free_g_comm(g_comm);
		free(h_table);
		//delete_mapper(index_mapper);
	 	//print_mapped_graph_yale(*g_yale, *index_mapper);
		delete_mapper(index_mapper);
		free_g_yale(g_yale);
		ds = 10;
		break;
	case 11:
		top1 ();
		ds = 3;
		g_comm = malloc(sizeof(graph_community_t));
		*g_comm = init_graph_community_sorted_neighbors(nomFichTree, nomFichTxt, oriented, h_table, g_param);//graphe taken from txt file are oriented, so oriented<-true
		index_mapper = do_index_mapping_sorted_sub_comm_2(*g_comm, *h_table);
		g_bloc = comm2bloc(*g_comm, index_mapper);
		bloc2txt(*g_bloc, nomFichTxt, "11_sorted_neigh-numbering_sub-comm");

	 	//print_graph_bloc(*g_bloc);
		free_h_table(h_table);
		free(h_table);
		free_g_param(g_param);
		free_g_comm(g_comm);
	//	delete_mapper(index_mapper);
	 	//print_mapped_graph_bloc(*g_bloc, *index_mapper);
		delete_mapper(index_mapper);
		free_g_bloc(g_bloc);
		ds = 11;
		break;
	case 12:
		top1 ();
		ds = 4;
		g_comm = malloc(sizeof(graph_community_t));
		*g_comm = init_graph_community_sorted_neighbors(nomFichTree, nomFichTxt, oriented, h_table, g_param);//graphe taken from txt file are oriented, so oriented<-true
		index_mapper = do_index_mapping_sorted_sub_comm_2(*g_comm, *h_table);
		g_yale = comm2yale_2(*g_comm, index_mapper, *g_param);
		yale2txt(*g_yale, nomFichTxt, "12_sorted_neigh-numbering_sub-comm");

	 	//print_graph_yale(*g_yale);
		free_h_table(h_table);
		free(h_table);
		free_g_param(g_param);
		free_g_comm(g_comm);
		//delete_mapper(index_mapper);
	 	//print_mapped_graph_yale(*g_yale, *index_mapper);
		delete_mapper(index_mapper);
		free_g_yale(g_yale);
		ds = 12;
		break;
	case 13:
		top1 ();
		ds = 3;
		g_comm = malloc(sizeof(graph_community_t));
		*g_comm = init_graph_community_sorted_neighbors(nomFichTree, nomFichTxt, oriented, h_table, g_param);//graphe taken from txt file are oriented, so oriented<-true
		index_mapper = do_index_mapping_sorted_sub_comm(*g_comm, *h_table);
		g_bloc = comm2bloc(*g_comm, index_mapper);
		bloc2txt(*g_bloc, nomFichTxt, "13_sorted_neigh-numbering_comm_and_sub-comm");

	 	//print_graph_bloc(*g_bloc);
		free_h_table(h_table);
		free(h_table);
		free_g_param(g_param);
		free_g_comm(g_comm);
	//	delete_mapper(index_mapper);
	 	//print_mapped_graph_bloc(*g_bloc, *index_mapper);
		delete_mapper(index_mapper);
		free_g_bloc(g_bloc);
		ds = 13;
		break;
	case 14:
		top1 ();
		ds = 4;
		g_comm = malloc(sizeof(graph_community_t));
		*g_comm = init_graph_community_sorted_neighbors(nomFichTree, nomFichTxt, oriented, h_table, g_param);//graphe taken from txt file are oriented, so oriented<-true
		index_mapper = do_index_mapping_sorted_sub_comm(*g_comm, *h_table);
		g_yale = comm2yale_2(*g_comm, index_mapper, *g_param);
		yale2txt(*g_yale, nomFichTxt, "14_sorted_neigh-numbering_comm_and_sub-comm");

	 	//print_graph_yale(*g_yale);
		free_h_table(h_table);
		free(h_table);
		free_g_param(g_param);
		free_g_comm(g_comm);
		//delete_mapper(index_mapper);
	 	//print_mapped_graph_yale(*g_yale, *index_mapper);
		delete_mapper(index_mapper);
		free_g_yale(g_yale);
		ds = 14;
		break;
	default:
		top1 ();
		printf("Unknown data structure, choose between 1 and 14!!!");
	}

	top2 () ;
	unsigned long temps = cpu_time();
	printf("\ntime = %ld.%03ldms, time_build_path = %ld.%03ldms, time_neig = %ld.%03ldms\n", temps/1000, temps%1000,time_build_path/1000,time_build_path%1000, time_neig/1000, time_neig%1000);

	sprintf(nom_fich,"katz_execution_time.txt");
        FILE* time_file = fopen(nom_fich, "a");
        fprintf(time_file,"%s\t%2f\t%d\t%d\t%d\t%ld.%03ld\t%ld.%03ld\t%ld.%03ld\n",nomFichGml,*beta,max_l,ds,nb_thread,temps/1000,temps%1000,time_build_path/1000,time_build_path%1000,time_neig/1000,time_neig%1000);
	//free(time_file);
	//printf("avant ord = %ld, après ord = %ld, gain = %ld\n",cost1/2,cost2/2, cost1/2-cost2/2);
	free(beta);
	free(nomFichTxt);free(nomFichGml);free(nomFichTree);
	return 0;
}

void bloc2txt(graph_bloc_t G_bloc, char* file_name, char* optimization)
{
	FILE* txt_file = NULL;
	char* reslt[len_string],*sep={" "};
	int orig, extr, ind_node, nb_node = G_bloc.nb_node, i, j, degree, ind_ia;
	char* resl_split[100], nom_fich[200];

	printf("%s\n",file_name);

	str_split(file_name,".",resl_split);
	sprintf(nom_fich,"%s_bloc_%s.txt",resl_split[0],optimization);
	remove(nom_fich);
        txt_file = fopen(nom_fich, "a");

	printf("n = %d\n", nb_node);

        for(i = 0; i < nb_node; i++)
	{

		degree  = G_bloc.node_table[i].degree;
		orig = G_bloc.node_table[i].node_id;
		printf("%d<%d>:", orig, degree);

		for(j = 0; j < degree; j++)
		{
			extr = G_bloc.node_table[i].neighbor_bloc[j];
			printf("%d ",extr);
			fprintf(txt_file,"%d %d\n",orig,extr);
		}
		printf("\n");
	}
        fclose(txt_file);
}

void yale2txt(graph_yale_t G_yale, char* file_name, char* optimization)
{
	FILE* txt_file = NULL;
	char* reslt[len_string],*sep={" "};
	int orig, extr, ind_node, nb_node = G_yale.nb_node, i = 0, degree, ind_ia;
	char* resl_split[100], nom_fich[200];

	printf("%s\n",file_name);

	str_split(file_name,".",resl_split);
	sprintf(nom_fich,"%s_yale_%s.txt",resl_split[0],optimization);
	remove(nom_fich);
        txt_file = fopen(nom_fich, "a");

	printf("n = %d\n", nb_node);
        for(ind_node = 0; ind_node < nb_node; ind_node++)
	{
		degree  = G_yale.IA[ind_node+1] - G_yale.IA[ind_node];
		orig    = ind_node;
		printf("%d<%d>:", orig, degree);

		ind_ia = G_yale.IA[ind_node];
		for(i = 0; i < degree; i++)
		{
			printf("%d ",G_yale.JA[ind_ia+i]);
			extr = G_yale.JA[ind_ia+i];
			fprintf(txt_file,"%d %d\n",orig,extr);
		}
		printf("\n");
	}
        fclose(txt_file);
}

void gml2txt(char* file_name)
{
        char ch1[20], ch2[20], ch[max_char_read], *tmp = {"test"};
        FILE* graph_file = NULL,*txt_file = NULL;
        char* reslt[len_string],*sep={" "};
        int orig, extr, id;
        int min_node = INT_MAX, oriented;
	char* resl_split[100], nom_fich[200];


	printf("%s\n",file_name);
        graph_file = fopen(file_name, "r");
	oriented = is_directed_gml(file_name);

	str_split(file_name,".",resl_split);
	sprintf(nom_fich,"%s.txt",resl_split[0]);
	remove(nom_fich);
        txt_file = fopen(nom_fich, "a");

        if(graph_file!= NULL)
        {
                while(tmp)
                {
                        tmp = fgets(ch, max_char_read, graph_file);

                        if(strcmp(trim_space(ch),"node")==0)
                        {
                                id = -1;
                                while(strcmp(trim_space(ch),"]")!=0)
                                {
                                        tmp = fgets(ch, max_char_read, graph_file);
                                        str_split(ch,sep,reslt);
                                        if(strcmp(trim_space(reslt[0]),"id")==0)
                                        {
						if(atoi(reslt[1])<min_node)
                                                        min_node = atoi(reslt[1]);
                                        }
                                }

                        }
			if(strcmp(trim_space(ch),"edge")==0)
                        {
                                while(strcmp(trim_space(ch),"]")!=0)
                                {
                                        tmp = fgets(ch, max_char_read, graph_file);
                                        str_split(ch,sep,reslt);
                                        if(strcmp(trim_space(reslt[0]),"source")==0)
                                        {
                                                orig = atoi(reslt[1]) - min_node;
                                        }
                                        if(strcmp(trim_space(reslt[0]),"target")==0)
                                        {
                                                extr = atoi(reslt[1]) - min_node;
						fprintf(txt_file,"%d %d\n",orig,extr);
						if(!oriented)
							fprintf(txt_file,"%d %d\n",extr,orig);
                                        }
                                }
                        }

                }
                fclose(graph_file);
                fclose(txt_file);
        }

}

void txt2gml(char* file_name)
{
        char ch1[20], ch2[20], ch[max_char_read], *tmp = {"test"};
        FILE* graph_file = NULL,*gml_file = NULL;
        char* reslt[len_string],*sep={" "};
        plist src_list = empty(), dst_list = empty();
        int min_node = INT_MAX, max_node = 0, src, dst, i, nb_edge = 0;
	char* resl_split[100], nom_fich[200], *src_temp, *dst_temp;


        graph_file = fopen(file_name, "r");
	printf("%s\n",file_name);
	

	str_split(file_name,".",resl_split);
	sprintf(nom_fich,"%s.gml",resl_split[0]);
	remove(nom_fich);

        if(graph_file!= NULL)
        {
                tmp = fgets(ch, max_char_read, graph_file);
                while(tmp)
                {
			str_split(ch,sep,reslt);
                        if(strlen(trim_space(tmp))>0)
			{
				src_temp = malloc(sizeof(char)*30);
				dst_temp = malloc(sizeof(char)*30);
				//printf("%s %s\n",reslt[0],reslt[1]);
				strcpy(src_temp,trim_space(reslt[0]));
				strcpy(dst_temp,trim_space(reslt[1]));
				//printf("%s %s\n",src_temp,dst_temp);
				//printf("%s %s\n",src_temp,dst_temp);
				src_list = cons(src_temp,src_list);
				dst_list = cons(dst_temp,dst_list);
                                src = atoi(trim_space(reslt[0]));
                                dst = atoi(trim_space(reslt[1]));
				if(src < min_node)
					min_node = src;
				if(dst < min_node)
					min_node = dst;
				if(src > max_node)
					max_node = src;
				if(dst > max_node)
					max_node = dst;
				nb_edge++;
			}
		 tmp = fgets(ch, max_char_read, graph_file);
                }
                fclose(graph_file);
       }
        gml_file = fopen(nom_fich, "a");
	fprintf(gml_file,"Creator\"Messi's program from %s.txt file nb_node = %d, nb_edge = %d\"\ngraph\n[\ndirected 1\n",file_name,(max_node-min_node+1),nb_edge);
	for(i=min_node;i<=max_node;i++)
	{
		fprintf(gml_file,"  node\n");
		fprintf(gml_file,"  [\n");
		fprintf(gml_file,"    id %d\n",i);
		fprintf(gml_file,"  ]\n");
		
	}

	while(!is_empty(src_list))
	{
		src_temp = head(src_list);
		dst_temp = head(dst_list);
		fprintf(gml_file,"  edge\n");
		fprintf(gml_file,"  [\n");
		fprintf(gml_file,"    source %s\n",src_temp);
		fprintf(gml_file,"    target %s\n",dst_temp);
		fprintf(gml_file,"  ]\n");
		src_list = tail(src_list);
		dst_list = tail(dst_list);
	}
	fprintf(gml_file,"]");
	fclose(gml_file);
}
