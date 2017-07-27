//
//  cacheLayerThread.cpp
//  mimircache
//
//  Created by Juncheng Yang on 7/13/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//

#include "cacheLayerThread.hpp"

/** this module is important in thread contention/race 
 *  current implementation maintains a vector of request at each cache layer,
 *  and cache layer runs periodically, during the run, it selects the requests 
 *  having timestamps smaller than minimal timestamp and add into cache layer 
 *
 *  it would be ideal to restrict the size of vector and block running 
 *  cacheServer thread if the vector is full
 */


/** Synchronization between cacheServer and cacheLayer is implemented 
 *  in this module */

namespace akamaiSimulator {
    
    
    bool cacheServerReqQueue::_can_add_req(cache_line_t* cp){
        
        /* no synchronization between layers */
//        return this->dq.size() < this->max_queue_size;
        
        /** synchronization between layers with max_time_dfff
         *  equals SYNCHRONIZE_TIME_DIFF */ 
        return this->dq.size() < this->max_queue_size &&
        (this->server_id == this->min_ts_server_id.load() ||
         std::abs((long) (cp->real_time - this->min_ts.load())) < SYNCHRONIZE_TIME_DIFF);
    }
    
    
    void cacheServerReqQueue::add_request(cache_line_t *cp){
        
        std::unique_lock<std::mutex> ulock(this->mtx);
//        while (this->dq.size() >= this->max_queue_size){
//            this->condt.wait(ulock);
//        }
        this->condt.wait(ulock, std::bind(&cacheServerReqQueue::_can_add_req, this, cp));
        this->dq.push_back(cp);
    }
    
    
    cache_line_t* cacheServerReqQueue::get_request(){
        std::lock_guard<std::mutex> lg(this->mtx);
        if (this->dq.empty())
            return NULL;

        cache_line_t* cp = this->dq.front();
        this->dq.pop_front();
        this->condt.notify_one();
        return cp;
    }
    
    cache_line_t* cacheServerReqQueue::get_request(double max_ts){
        std::lock_guard<std::mutex> lg(this->mtx);
        if (this->dq.empty()){
//            verbose("return due to empty\n");
            return NULL;
        }
            
        cache_line_t* cp = this->dq.front();

        if (cp->real_time <= max_ts){
            this->dq.pop_front();
            this->condt.notify_one();
        }
        else{
//            verbose("return due to ts %ld, %lf, size %lu\n", cp->real_time, max_ts, this->dq.size());
            cp = NULL;
        }
        return cp;
    }
    
    void cacheServerReqQueue::synchronize_time(double t, unsigned long min_ts_ind){
        this->min_ts_server_id.store(min_ts_ind);
        this->min_ts.store(static_cast<unsigned long>(t));
        this->condt.notify_one();
    }
    
    
    
    
    
    
    void cacheLayerThread::init(){
        this->layer_id = -1;
        this->minimal_timestamp = -1;
        this->last_log_output_time = 0; 
        this->minimal_ts_server_ind = -1;
        this->all_server_finished = false;
    }
    
    
    cacheLayerThread::cacheLayerThread(cacheLayer *cache_layer){
        this->init();
        this->layer_id = cache_layer->get_layer_id();
        this->cache_layer = cache_layer;
        this->num_of_servers = (unsigned long) cache_layer->get_num_server();
        this->cache_server_timestamps = new double[this->num_of_servers];

        
        this->cache_server_req_queue = new cacheServerReqQueue*[this->num_of_servers];
        for (unsigned long i=0; i<this->num_of_servers; i++){
            this->cache_server_req_queue[i] = new cacheServerReqQueue(MAX_SERVER_REQ_QUEUE_SIZE, i);
            this->cache_server_timestamps[i] = 0;
        }
        
//        this->req_pq = new std::priority_queue<
//            cache_line_t*, std::vector<cache_line_t*>, cp_comparator> ();

        
        
        for (unsigned long i=0; i<this->num_of_servers; i++)
            this->cache_server_timestamps[i] = 0;
    }
    
    cacheLayer* cacheLayerThread::get_cache_layer(){
        return this->cache_layer;
    }
    
    
    double cacheLayerThread::get_minimal_timestamp(){
        return this->minimal_timestamp;
    }
    
    cacheLayerStat* cacheLayerThread::get_layer_stat(){
        return this->cache_layer->get_layer_stat();
    }

    
    
//    void cacheLayerThread::run(unsigned int log_interval){
//        bool all_server_finished_local = false;
//        gboolean hit;
//        cacheLayer *next_layer = this->cache_layer->get_next_layer();
//        if (this->layer_id < NUM_CACHE_LAYERS && next_layer == NULL){
//            error_msg("missing next layer pointer at layer %d\n", this->layer_id);
//            abort();
//        }
//            
//        int i;
//        cache_line_t * cp;
//        
//        if (log_interval != 0){
//            mkdir(LOG_FOLDER, 0770);
//            this->log_file_loc = std::string(LOG_FOLDER) + std::string("/layer")
//                                    + std::to_string(this->layer_id);
//            this->log_filestream.open(log_file_loc);
//        }
//        
//        while (! this->all_server_finished){
//            this->mtx_add_req.lock();
//            if (!this->req_pq->empty()){
//                cp = this->req_pq->top();
//                while (! this->req_pq->empty() &&
//                       cp->real_time <= this->minimal_timestamp.load()){
//                    /* now add to current layer */
//                    hit = this->cache_layer->add_request(cp->cache_server_id, cp);
//                    
//                    /* logic for add missed request into next layer */
//                    if (!hit && this->layer_id < NUM_CACHE_LAYERS){
//                        next_layer->add_request(cp->cache_server_id, cp);
//                    }
//                    
//                    /* log stat */
//                    if (cp->real_time - this->last_log_output_time > log_interval){
//                        this->log_layer_stat((unsigned long) cp->real_time);
//                        this->last_log_output_time = (unsigned long) cp->real_time;
//                    }
//                    
//                    /* clean this cp and retrieve next cp */
//                    this->req_pq->pop();
//                    destroy_cacheline(cp);
//                    cp = this->req_pq->top();
//                }
//            }
//            this->mtx_add_req.unlock();
//            std::this_thread::sleep_for(std::chrono::milliseconds(200));
//            
//            
//            /** check whether all cache servers have finished their trace
//             *  ATTENTION: it might be better to have lock here */
//            if (this->req_pq->empty()){
//                all_server_finished_local = true;
//                for (i=0; i<this->num_of_servers; i++){
//                    if ( std::abs(LONG_MAX - (long) this->cache_server_timestamps[i]) > 1 ){
//                        all_server_finished_local = false;
//                        break;
//                    }
//                }
//                this->all_server_finished = all_server_finished_local;
//            }
//        }
//        
//        if (log_interval != 0){
//            this->log_filestream.close(); 
//        }
//    }

    
    void cacheLayerThread::run(unsigned int log_interval){
        bool all_server_finished_local = false;
        gboolean hit;
        cacheLayer *next_layer = this->cache_layer->get_next_layer();
        if (this->layer_id < NUM_CACHE_LAYERS && next_layer == NULL){
            error_msg("missing next layer pointer at layer %d\n", this->layer_id);
            abort();
        }
        
        unsigned long i;
        cache_line_t * cp;
        double min_ts;
        bool found_no_new_req;
        std::priority_queue<cache_line_t*, std::vector<cache_line_t*>, cp_comparator> req_pq;
        
        if (log_interval != 0){
            mkdir(LOG_FOLDER, 0770);
            this->log_file_loc = std::string(LOG_FOLDER) + std::string("/layer")
            + std::to_string(this->layer_id);
            this->log_filestream.open(log_file_loc);
            system("cp /home/jason/pycharm/mimircache/CExtension/bin/*.sh log/");
        }
        
        while (! this->all_server_finished){
            /** get all the req that have timestamp <= min_ts
             *  order them using priorityQueue and add to servers in the layer */
            min_ts = this->get_minimal_timestamp();
            
            if (min_ts+1 < 0.0001){
                // this means min_ts == -1, the initialization is not finished
                std::this_thread::sleep_for(std::chrono::seconds(2));
                continue;
            }
            
            found_no_new_req = true;
            for (i=0; i<this->num_of_servers; i++){
                cp = this->cache_server_req_queue[i]->get_request(min_ts);
                while (cp != NULL) {
                    req_pq.push(cp);
                    found_no_new_req = false;
                    cp = this->cache_server_req_queue[i]->get_request(min_ts);
                }
            }
            
            
            while (! req_pq.empty()){
                cp = req_pq.top();
                /* now add to current layer */
                hit = this->cache_layer->add_request(cp->cache_server_id, cp);
                    
                /* logic for add missed request into next layer */
                if (!hit && this->layer_id < NUM_CACHE_LAYERS){
                    next_layer->add_request(cp->cache_server_id, cp);
                }
                    
                /* log stat */
                if (cp->real_time - this->last_log_output_time > log_interval){
                    this->log_layer_stat((unsigned long) cp->real_time);
                    this->last_log_output_time = (unsigned long) cp->real_time;
                }
                    
                /* clean this cp and retrieve next cp */
                req_pq.pop();
                destroy_cacheline(cp);
                cp = req_pq.top();
            }        
        
            
            /** check whether all cache servers have finished their trace
             *  ATTENTION: it might be better to have lock here */
            if (found_no_new_req){
                all_server_finished_local = true;
                for (i=0; i<this->num_of_servers; i++){
                    if ( std::abs(LONG_MAX - (long) this->cache_server_timestamps[i]) > 1 ){
                        all_server_finished_local = false;
                        break;
                    }
                }
                this->all_server_finished = all_server_finished_local;
                if (!all_server_finished_local){
//                    warning("no new req\n");
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }
            }
        }
        
        if (log_interval != 0){
            this->log_filestream.close();
        }
    }


    
    /** the difference between cacheLayerThread::add_request and
     *  cacheLayer::add_request is that
     *  the first one adds the request into
     *  a priority queue to buffer the requests from different cache servers,
     *  it will not be added directly into caches, 
     *  instead, the priorityqueue is scanned periodically
     *  to add possible requests (timestamps < the timestamp of the slowest server) into caches.
     *  the second one directly addes the request into caches. 
     *  
     *  \param  cache_server_id  where the cache request comes from
     *  \param  cp               cache request
     */
//    void cacheLayerThread::add_request(unsigned long cache_server_id, cache_line_t *cp){
//        this->cache_server_timestamps[cache_server_id] = cp->real_time;
//        
//        /** copy is done outside of mtx to reduce the time of lock
//         *  if the function failed, then this memory will not be released */
//        cache_line_t * const cp_new = copy_cache_line(cp);
//        cp_new->cache_server_id = cache_server_id;
//        
//        std::lock_guard<std::mutex> lg(this->mtx_add_req);
//        
//        
//        
//        if (this->minimal_ts_server_ind == (long) cache_server_id ||
//            this->minimal_ts_server_ind == -1){
//            /* now need to find out the new minimal ts */
//            this->minimal_ts_server_ind =
//                (int) (std::min_element(this->cache_server_timestamps,
//                                        this->cache_server_timestamps +
//                                        this->num_of_servers)
//                       - this->cache_server_timestamps);
//
//            this->minimal_timestamp.store(this->cache_server_timestamps[this->minimal_ts_server_ind]);
//        }
//        std::cout.precision(8);
//        this->req_pq->push(cp_new);
//    }

    
    void cacheLayerThread::add_request(unsigned long cache_server_id, cache_line_t *cp){
        cache_line_t * const cp_new = copy_cache_line(cp);
        cp_new->cache_server_id = cache_server_id;

        
        std::unique_lock<std::mutex> ulock(this->mtx_add_req);

        this->cache_server_timestamps[cache_server_id] = cp->real_time;
        
        /** copy is done outside of mtx to reduce the time of lock
         *  if the function failed, then this memory will not be released */
        
        
        
        /** this is not exactly mt-safe
         *  reading from cache_server_timestamps can be dangerous */
        if (this->minimal_ts_server_ind == (long) cache_server_id ||
            this->minimal_ts_server_ind == -1){

            /* now need to find out the new minimal ts */
            this->minimal_ts_server_ind = std::min_element(this->cache_server_timestamps,
                                    this->cache_server_timestamps +
                                    this->num_of_servers)
                   - this->cache_server_timestamps;

            this->minimal_timestamp = this->cache_server_timestamps[this->minimal_ts_server_ind];


//            verbose("new min ts %lf at index %lu, from %lu, ts %lf %lf\n", min_ts,
//                    min_ts_ind, cache_server_id,
//                    this->cache_server_timestamps[0], this->cache_server_timestamps[1]);

            // synchronize cache servers
            for (unsigned long i=0; i<this->num_of_servers; i++){
                this->cache_server_req_queue[i]->synchronize_time(this->minimal_timestamp,
                                                                  this->minimal_ts_server_ind);
                
            }
        }
        
        /* unlock before adding req, because it is blocking */
        ulock.unlock();
        
        /* this needs to be after time synchronization */
        this->cache_server_req_queue[cache_server_id]->add_request(cp_new);
    }

    
    
    void cacheLayerThread::cache_server_finish(unsigned long cache_server_id){
        this->cache_server_timestamps[cache_server_id] = (double) (LONG_MAX);
        /** when minimal_ts_server_ind is current server, then it won't change
         *  in other threads, because only current server can change it, 
         *  threrefore it is thread-safe */
        if (this->minimal_ts_server_ind == (long) cache_server_id){
            /* now need to find out the new minimal ts */
            this->minimal_ts_server_ind = std::min_element(this->cache_server_timestamps,
                                    this->cache_server_timestamps +
                                    this->num_of_servers)
                                    - this->cache_server_timestamps;
            this->minimal_timestamp = this->cache_server_timestamps[this->minimal_ts_server_ind]; 

//            verbose("trace finish, new min ts %lf from %lu\n", min_ts, cache_server_id);
  
            // synchronize cache servers
            for (unsigned long i=0; i<this->num_of_servers; i++)
                this->cache_server_req_queue[i]->synchronize_time(this->minimal_timestamp,
                                                                  this->minimal_ts_server_ind);
        }
    }
    
    
    void cacheLayerThread::log_layer_stat(unsigned long layer_time) {
        log_filestream << layer_time << "\t" <<
            cacheLayer::build_stat_str_short(this->cache_layer->layer_stat);
    }
    
    
    
    cacheLayerThread::~cacheLayerThread(){
        delete [] this->cache_server_timestamps;
        for (unsigned int i=0; i<this->num_of_servers; i++)
            delete this->cache_server_req_queue[i];
        delete [] this->cache_server_req_queue;
//        delete this->req_pq;
    }
}
