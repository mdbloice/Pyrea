/**
 * @file HC_fused_cpp_opt6.cpp
 *
 * @brief Implements the HC Fusion algorithm
 *
 * @author Andrei Voicu-Spineanu, Marcus D. Bloice, Bastian Pfeifer
 * Modified for Pyrea by: Marcus D. Bloice and Bastian Pfeifer
 *
 */
#include <iostream>
#include <math.h>
#include <chrono>
#include <vector>
#include "HC_fused_cpp_opt6.hpp"
using namespace std;

// Function which calculates the mean value from a matrix
double getmean(int i,int j, int *mat, int elem, int *obj_mat, int *obj_sizes, int n_patients){
    double accum = 0;

    for(int k=0;k<obj_sizes[i];k++){ // select the line i of obj_mat and iterate through its elements
        for(int l=0;l<obj_sizes[j];l++){ // select the line j of obj_mat and iterate through its elements
            accum += mat[elem*(n_patients)*n_patients + obj_mat[i*n_patients + k]*n_patients + obj_mat[j*n_patients + l]];
        }
    }
    accum=accum/(obj_sizes[i]*obj_sizes[j]);

    return accum;
}

// Function that calculates the similarity matrix - distances
void HC_fused_calc_distances_cpp(int *obj, int *MAT, int *matAND, int *obj_sizes, double *DISTANCES, double *mat_distances,int n_elems,int n_patients, int n_clusters, int col_nr){
    //reset mat_distances
    for(int i=0;i<col_nr;i++){
        mat_distances[i] = 0;
    }

    int dist_index = 0;

    for(int elem=0;elem<n_elems;elem++){
        for(int i=0;i<n_clusters-1;i++){
            for(int j=i+1;j<n_clusters;j++){
                mat_distances[dist_index] = getmean(i,j,MAT,elem,obj,obj_sizes,n_patients);
                dist_index++;
            }
        }

        for(int j=0;j<col_nr;j++){
            DISTANCES[elem*(col_nr)+j] = mat_distances[j];
        }
        dist_index = 0; // Reset the dist_index for the next matrix
    };

    dist_index = 0;
    for(int i=0;i<n_clusters-1;i++){
        for(int j=i+1;j<n_clusters;j++){
            mat_distances[dist_index] = getmean(i,j,matAND,0,obj,obj_sizes,n_patients);
            dist_index++;
        }
    }

    for(int j=0;j<col_nr;j++){
        DISTANCES[n_elems*(col_nr)+j] = mat_distances[j];//last line from DISTANCES
    }
}

// Function that returns indices values of elements equal to the max of distances
void which_vec_mat(double *distances, int *res_vec, int *res_mat, int &ids_valid_size, int n_clusters,int n_elems,int col_nr){

    // Calculate distances valid size:
    int distances_valid_size = n_clusters*(n_clusters-1)/2; // Based on combinations of n taken by 2
    double max_el = distances[0];
    int res_counter=0;

    for(int i=0;i<distances_valid_size;i++){ //increments column
        for(int j=0;j<(n_elems+1)*col_nr ;j=j+col_nr){ //increments line
            if(distances[i+j]>max_el){
                //reset res_counter
                res_counter=0;
                //update the max
                max_el = distances[i+j];
                //add indices
                res_vec[res_counter] = (i+1) + j/col_nr + i*n_elems;
                res_mat[2*res_counter]= j/col_nr + 1;
                res_mat[2*res_counter+1]= i+1;
                res_counter++;
            }else if(distances[i+j] ==max_el){
                //add indices
                res_vec[res_counter] = (i+1) + j/col_nr + i*n_elems;
                res_mat[2*res_counter]= j/col_nr + 1;
                res_mat[2*res_counter+1]= i+1;
                res_counter++;
            }
        }
    }
    ids_valid_size = res_counter;
}

// Function that returns the line number of elements from ids2 which are equal to the no. of rows from distances
void which_is3(int *ids2,int ids_valid_size,int dist_size, int *is3_res, int &is3_valid_size){
    int is3_counter = 0;
    for(int i=0;i<2*ids_valid_size;i=i+2){
        if (ids2[i]==dist_size){
            is3_res[is3_counter] = i/2; // /2 in order to get only the lines from ids2
            is3_counter++;
        }
    }
    is3_valid_size = is3_counter;
}

// Function used to get a single number out of a vector
int get_sample(int *vec,bool randomize, int vec_valid_size){ //also add valid vector size as argument
    if(randomize == false){
        return vec[0];
    }else{
        return vec[(rand()%vec_valid_size)];
    }
}

// Function that updates map_info_pair, indicating which patients clusters will merge
void get_ij(int index,int obj_size, int *map_info_pair){
    int col_elems = obj_size-1;
    int nr_elems = obj_size-1;
    int sum_col_elems = 0;
    bool found = false;

    if(index<obj_size-1){ //first column
        map_info_pair[0]=0;
        map_info_pair[1]=index+1;
        return;
    }

    for(int i=1;i<obj_size-1;i++){
        col_elems--;
        nr_elems +=col_elems;
        sum_col_elems += col_elems;
        if(index<nr_elems && found == false){
            map_info_pair[0]=i;
            map_info_pair[1]=index-sum_col_elems+1;
            found = true;
            return;
        }
    }
}

// Function to call from Python
int* HC_fused_cpp_opt6(int* MAT_array, int n_cluster_arrays, int cluster_size, int n_iter)
{
    // Convert the 2D array into a 2D vector
    vector<vector<int>> MAT(n_cluster_arrays, vector<int>(cluster_size));

    for(int i = 0; i < n_cluster_arrays; i++){
        for(int j = 0; j < cluster_size; j++){
            MAT[i][j] = MAT_array[(i * cluster_size) + j];
        }
    }

    // Memory allocations
    // HC_fused function
    int n_patients = sqrt(MAT[0].size());
    int *obj = (int*)malloc(sizeof(int)*n_patients*n_patients);
    int *obj_sizes_initial = (int*)malloc(sizeof(int)*n_patients);
    for(int i=0;i<n_patients;i++){
        obj[i*n_patients]=i;
        obj_sizes_initial[i]=1;
        for(int j=1;j<n_patients;j++){ //skip j=0 (first column)
            obj[i*n_patients+j]=0;
        }
    }

    int n_pat_sqr = MAT[0].size();
    int n_elems = MAT.size();

    // Input argument pointer
    int *MAT_p = (int*)malloc(sizeof(int)*n_elems*n_pat_sqr);
    int index = 0;
    for(int i=0;i<n_elems;i++){
        for(int j=0;j<n_pat_sqr;j++){
            MAT_p[index] = MAT[i][j];
            index++;
        }
    }

    int *matAND = (int*)malloc(sizeof(int)*n_pat_sqr);
    for(int i=0;i<n_pat_sqr;i++){
        matAND[i] = MAT[0][i];
    }
    for(int elem=1;elem<n_elems;elem++){
        for(int i=0;i<n_pat_sqr;i++){
            matAND[i] = matAND[i] && MAT[elem][i];
        }
    }

    int *NETWORK = (int*)malloc(sizeof(int)*n_patients*n_patients);
    for(int i=0;i<n_patients*n_patients;i++){
        NETWORK[i]=0;
    }

    int *obj_dyn = (int*)malloc(sizeof(int)*n_patients*n_patients);
    for(int i=0;i<n_patients*n_patients;i++){
        obj_dyn[i] = 0;
    }

    int *obj_sizes = (int*)malloc(sizeof(int)*n_patients);
    for(int i=0;i<n_patients;i++){
        obj_sizes[i]=0;
    }

    int *map_info_pair = (int*)malloc(sizeof(int)*2);
    map_info_pair[0]=0;
    map_info_pair[1]=0;
    int id_min=0;

    // Calc_distances function allocations
    int col_nr = n_patients*(n_patients-1)/2;
    double *distances = (double*)malloc(sizeof(double)*(n_elems+1)*col_nr);
    for(int i=0;i<(n_elems+1)*col_nr;i++){
        distances[i] = 0;
    }

    double *mat_distances = (double*)malloc(sizeof(double)*col_nr);
    for(int i=0;i<col_nr;i++){
        mat_distances[i]=0;
    }

    // which_vec_mat function allocations
    int distances_total_size = (n_elems+1)*col_nr;
    int *res_vec = (int*)malloc(sizeof(int)*distances_total_size);
    for(int i=0;i<distances_total_size;i++){
        res_vec[i]=0;
    }
    int *res_mat = (int*)malloc(sizeof(int)*distances_total_size*2);
    for(int i=0;i<distances_total_size*2;i++){
        res_mat[i]=0;
    }

    int ids_valid_size = 0;

    // which_is3 function allocations
    int *is3 = (int*)malloc(sizeof(int)*distances_total_size);
    for(int i=0;i<distances_total_size;i++){
        is3[i]=0;
    }
    int is3_valid_size = 0;


    for(int xx=0;xx<n_iter;xx++){

        // Reinitialize obj_dyn
        for(int i=0;i<n_patients*n_patients;i++){
            obj_dyn[i]=obj[i];
        }
        // Reinitialize n_clusters
        int n_clusters = n_patients;
        // Reinitialize obj_sizes
        for(int i=0;i<n_patients;i++){
            obj_sizes[i]=obj_sizes_initial[i];
        }

        for(int obj_iter=1; obj_iter<n_patients;obj_iter++){

            HC_fused_calc_distances_cpp(obj_dyn, MAT_p, matAND, obj_sizes,distances, mat_distances,n_elems,n_patients, n_clusters, col_nr);

            which_vec_mat(distances,res_vec, res_mat, ids_valid_size, n_clusters, n_elems, col_nr); // Updates res_vec and res_mat

            which_is3(res_mat, ids_valid_size, n_elems+1, is3, is3_valid_size); // Updates is3


            if(is3_valid_size!=0){
                if(is3_valid_size>1){
                    id_min = res_vec[get_sample(is3,true,is3_valid_size)]; // True for random number, false for the first number from the vector
                }else{ // Only one element in is3
                    id_min = res_vec[is3[0]];
                }
            }else{ // is3 is empty
                if(ids_valid_size>1){
                    id_min = get_sample(res_vec,true, ids_valid_size); // True for random number, false for the first number from the vector
                }else{ // Only one element in ids
                    id_min = res_vec[0];
                }
            }
            id_min--;

            get_ij(id_min/(n_elems+1), n_clusters , map_info_pair);//updates map_info_pair

            // Insert and adjust the size of obj_sizes
            for(int i=0;i<obj_sizes[map_info_pair[1]];i++){ // Based on the number of elements in the soon-to-be erased vector
                obj_dyn[map_info_pair[0]*n_patients + obj_sizes[map_info_pair[0]] +i] = obj_dyn[map_info_pair[1]*n_patients + i];
            }
            obj_sizes[map_info_pair[0]] += obj_sizes[map_info_pair[1]];

            for(int elem=map_info_pair[1];elem<n_clusters-1;elem++){
                for(int i=0;i<n_patients;i++){
                    obj_dyn[elem*n_patients + i] = obj_dyn[(elem+1)*n_patients + i]; // Translate the patients starting from map_info_pair[1]
                }
            }
            // Last vector from obj_dyn:
            for(int i=0;i<n_patients;i++){
                obj_dyn[(n_clusters-1)*n_patients + i] = 0;
            }


            // obj_sizes
            for(int elem=map_info_pair[1];elem<n_clusters-1;elem++){
                obj_sizes[elem] = obj_sizes[elem+1];
            }
            // Last element from obj_sizes:
            obj_sizes[n_clusters-1] = 0;

            n_clusters--; // Added to take into account the decreasing size of obj_dyn

            for(int obj_index=0;obj_index<n_clusters;obj_index++){ //Go through the vectors of obj_dyn
                if( obj_sizes[obj_index]>1){
                    for(int i=0;i<obj_sizes[obj_index];i++){
                        for(int j=0; j<obj_sizes[obj_index];j++){
                            NETWORK[obj_dyn[obj_index*n_patients + i]*n_patients + obj_dyn[obj_index*n_patients + j]]++;
                        }
                    }
                }
            } // end of inner for loop
        } // end of while loop
    } // end of outer for loop

    /*
     * Remove. We do not need to convert to vectors for returning to Python.
    //NETWORK got updated
    int network_index = 0;
    vector<vector<int>> NETWORK_RET (n_patients,vector<int>(n_patients));
    for(int i=0;i<n_patients;i++){
        for(int j=0;j<n_patients;j++){
            NETWORK_RET[i][j] = NETWORK[network_index];
            network_index++;
        }
    }
    */

    return NETWORK;

}
