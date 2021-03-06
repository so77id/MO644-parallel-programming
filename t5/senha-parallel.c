#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>

int STEPS=10;
int MAX_TRY = 500000;

char finalcmd[300] = "unzip -P%d -t %s 2>&1";

/*************************************************************************************************
The algorithm consists in using a thread that is producing numbers in a queue, which is being consumed by consumer threads, which are trying to break the encryption of the .zip

The important thing of this algorithm is in the production of the queue, this because the numbers can be produced in any order, for the case that was decided, the numbers are generated by blocks and it runs as if it were an array, The step is 10 the numbers are generated, 0, 10, 20, 30, 40, ..., 500000 and then 1, 11, 21, ... ETC.
**************************************************************************************************/


/**************************************
INIT QUEUE IMPLEMENTATION
***************************************/

struct node {
    int value;
    struct node *next;
};

struct queue {
    struct node *head;
    struct node *tail;
    int n_nodes;
};

// enqueue to Queue
void enqueue(struct queue *q, int value) {
    struct node *new = malloc(sizeof(struct node));
    if(new == NULL) return;
    new->value = value;
    new->next = NULL;

    if(q->tail == NULL){
        q->tail = q->head = new;
    } else {
        q->tail->next = new;
        q->tail = new;
    }
}

int empty(struct queue *q) {
    if(q->head) return 1;
    else return 0;
}


int dequeue(struct queue *q) {
    if(q->head == NULL) return -1;
    struct node *tmp = q->head;
    q->head = tmp->next;
    int value = tmp->value;
    free(tmp);
    return value;
}

// Return new pointer to queue
struct queue* init_queue(){
    struct queue *q = malloc(sizeof(struct queue));
    q->head = q->tail = NULL;
    return q;
}


// Delete queue
void delete_queue(struct queue* q) {
    struct node *tmp;
    while(q->head != NULL) {
        tmp = q->head;
        q->head = tmp->next;
        free(tmp);
    }
    free(q);
}

void print_queue(struct queue *q) {
    printf("Printing: \n");
    for(struct node *iter = q->head; iter != NULL; iter = iter->next) {
        printf("%d ", iter->value);
    }
    printf("\n");
}

/**************************************
END QUEUE IMPLEMENTATION
***************************************/


//FILE *popen(const char *command, const char *type);
// Function to calculate time
double rtclock()
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


/*******************************
INIT Threads structs and workers
*******************************/

// Structure used to share data between threads
struct worker_data
{
    struct queue *q;
    int queue_finished;
    int result;
    char *filename;
    pthread_mutex_t *mutex;
};


// Worker charge of inserting to queue
void *queue_worker(void *arg){
    // Get data
    struct worker_data *wd = arg;

    int i, j;
    int max_try =  MAX_TRY + (MAX_TRY % STEPS);
    int segments = max_try / STEPS;

    // Insert in queue numbers in special order by steps
    for (i = 0; i < STEPS; ++i)
    {
        for (j = 0; j < segments; ++j)
        {
            //printf("encolando: %d\n", j * STEPS + i);
            enqueue(wd->q, j * STEPS + i);
        }
    }

    for (int i = 0; i < MAX_TRY; ++i)
    {
        //printf("encolando: %d\n", i);
        enqueue(wd->q, i);
    }
    wd->queue_finished = 1;
}

// Worker charge of check if password works
void *broke_worker(void *arg){
    // Get data
    struct worker_data *wd = arg;
    struct queue *q = wd->q;

    // Create private variables
    FILE * fp;
    char cmd[400];
    char ret[200];
    int value;

    while(1) {
        // if result was found, left loop
        if(wd->result != -1) break;
        // if queue is empty and queue_worker finish his work, left loop
        if(!empty(q) && wd->queue_finished) break;
        // if queue is empty and queue_worker not finish his work, continue to next iteration
        if(!empty(q)) continue;

        // ADQUIRE MUTEX QUEUE
        pthread_mutex_lock(wd->mutex);
        // Get net try
        value = dequeue(q);
        pthread_mutex_unlock(wd->mutex);
        // RELESE MUTEX QUEUE

        //Create cmd string
        sprintf((char*)&cmd, finalcmd, value, wd->filename);
        //printf("thread: %u, Comando a ser executado: %s \n", self_id, cmd);
        // Open File
        fp = popen(cmd, "r");
        while (!feof(fp)) {
          //Get data from file
          fgets((char*)&ret, 200, fp);
          //File data from file is equal to ok, password works
          if (strcasestr(ret, "ok") != NULL) {
            //printf("ENCONTRE EL RESULTADO %d\n", value);
            // Save result and warn to other threads that work is finished
            wd->result = value;
          }
        }

        //close file
        pclose(fp);
    }
}
/*******************************
END Threads structs and workers
*******************************/



int main ()
{
    int n_threads;

    char filename[100];
    double t_start, t_end;

    int i;

    // Scan data
    scanf("%d", &n_threads);
    scanf("%s", filename);

    // Creating queue
    struct queue *q = init_queue();

    // Creating threads
    pthread_t *threads = malloc((n_threads+1)* sizeof(pthread_t));

    // Creating mutex
    pthread_mutex_t mutex;

    // Creating threads data
    struct worker_data threads_data;
    threads_data.q = init_queue();
    threads_data.result = -1;
    threads_data.queue_finished = 0;
    threads_data.filename = filename;
    threads_data.mutex = &mutex;

    // Init mutex
    pthread_mutex_init (threads_data.mutex, NULL);

    // start proccess
    t_start = rtclock();
    // Create queue thread
    pthread_create(&threads[0], NULL, queue_worker, &threads_data);

    // Create broker threads
    for (i = 1; i <= n_threads; ++i)
    {
        pthread_create(&threads[i], NULL, broke_worker, &threads_data);
    }

    // Join to each thread for continue
    for (i = 0; i <= n_threads; ++i)
    {
        pthread_join(threads[i], NULL);
    }
    t_end = rtclock();

    // Free memory
    free(threads);
    delete_queue(q);
    pthread_mutex_destroy(threads_data.mutex);


    // print results
    fprintf(stdout, "%d\n", threads_data.result);
    fprintf(stdout, "%0.6lf\n", t_end - t_start);

    return(0);
}






/*************





Serial
arq6.in 245999 11096.664200

Parallel
arq1.in 10000 80.669087
arq2.in 100000 947.541681
arq3.in 450000 606.342429
arq4.in 310000 92.065449
arq5.in 65000 13.298806
arq6.in 245999 49.467526

**************/