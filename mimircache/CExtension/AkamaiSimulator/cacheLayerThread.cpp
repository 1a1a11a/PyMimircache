//
//  cacheLayerThread.cpp
//  mimircache
//
//  Created by Juncheng Yang on 7/13/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//

#include "cacheLayerThread.hpp"
#include <exception>


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
    
    cacheServerReqQueue::cacheServerReqQueue(unsigned long max_queue_size,
                                             unsigned long server_id){
        this->server_id = server_id;
        this->max_queue_size = max_queue_size;
        this->min_ts_server_id.store(0);
        this->min_ts.store(0);
    }
    
    
    
    bool cacheServerReqQueue::_can_add_req(cache_line_t* cp){
        
#ifndef TIME_SYNCHRONIZATION
        /* no synchronization between layers */
        return this->dq.size() < this->max_queue_size;
#else
        /** synchronization between layers with max_time_dfff
         *  equals SYNCHRONIZATION_TIME_DIFF */
//        debug("server %lu, real time %ld, %d (%lu, %lu): %d: %d (%ld, %d)\n",
//              this->server_id, cp->real_time,
//              this->dq.size() < this->max_queue_size,
//              this->dq.size(), this->max_queue_size,
//              this->server_id == this->min_ts_server_id.load(),
//              std::abs((long) (cp->real_time - this->min_ts.load())) < SYNCHRONIZATION_TIME_DIFF,
//              std::abs((long) (cp->real_time - this->min_ts.load())), SYNCHRONIZATION_TIME_DIFF);
        
        return this->dq.size() < this->max_queue_size &&
        (this->server_id == this->min_ts_server_id.load() ||
         std::abs((long) (cp->real_time - this->min_ts.load())) < SYNCHRONIZATION_TIME_DIFF);
#endif
        
    }
    
    
    void cacheServerReqQueue::add_request(cache_line_t *cp, bool real_add){
        /** a server thread will be blocked here, if one the following conditions are met:
         *  a. stored cache_line has reached max_queue_size,
         *  b. current server runs too fast and its time difference with the slowest server
         *     is larger than SYNCHRONIZATION_TIME_DIFF
         */
        this->current_ts = (unsigned long) cp->real_time;
        std::unique_lock<std::mutex> ulock(this->mtx);
        this->condt.wait(ulock, std::bind(&cacheServerReqQueue::_can_add_req, this, cp));
//        debug("passed check, server %lu, real time %ld\n", this->server_id, cp->real_time);
        if (real_add)
            this->dq.push_back(cp);
    }
    
    
    cache_line_t* cacheServerReqQueue::get_request(){
        std::lock_guard<std::mutex> lg(this->mtx);
        if (this->dq.empty()){
            this->condt.notify_one();
            return NULL;
        }
        cache_line_t* cp = this->dq.front();
        this->dq.pop_front();
        this->condt.notify_one();
        return cp;
    }
    
    cache_line_t* cacheServerReqQueue::get_request(double max_ts){
        std::lock_guard<std::mutex> lg(this->mtx);
        if (this->dq.empty()){
            this->condt.notify_one();
            return NULL;
        }
        
        cache_line_t* cp = this->dq.front();
        
        if (cp->real_time <= max_ts){
            this->dq.pop_front();
        }
        else{
            this->condt.notify_one();
            cp = NULL;
        }
        return cp;
    }
    
    size_t cacheServerReqQueue::get_size(){
        return this->dq.size();
    }
    
    unsigned long cacheServerReqQueue::get_min_ts(){
        return this->min_ts.load();
    }
    
    unsigned long cacheServerReqQueue::get_min_ts_server_id(){
        return this->min_ts_server_id.load();
    }
    
    
    void cacheServerReqQueue::synchronize_time(double t, unsigned long min_ts_ind){
        this->min_ts_server_id.store(min_ts_ind);
        this->min_ts.store(static_cast<unsigned long>(t));
        this->condt.notify_one();
    }
    
    void cacheServerReqQueue::queue_size_change(){
//        debug("server %lu queue size change notified\n", this->server_id);
        this->condt.notify_all();
    }
    
    
    
    
    void cacheLayerThread::init(){
        this->layer_id = -1;
        this->minimal_timestamp.store(0);
        this->last_log_output_time = 0;
        this->minimal_ts_server_ind.store(0);
        this->all_server_finished = false;
    }
    
    
    cacheLayerThread::cacheLayerThread(cacheLayer *cache_layer){
        this->init();
        this->layer_id = cache_layer->get_layer_id();
        this->cache_layer = cache_layer;
        this->num_of_servers = (unsigned long) cache_layer->get_num_server();
        //        this->cache_server_timestamps = new double[this->num_of_servers];
        this->cache_server_timestamps = new std::atomic<double>[this->num_of_servers];
        
        
        this->cache_server_req_queue = new cacheServerReqQueue*[this->num_of_servers];
        for (unsigned long i=0; i<this->num_of_servers; i++){
            this->cache_server_req_queue[i] = new cacheServerReqQueue(MAX_SERVER_REQ_QUEUE_SIZE, i);
            //            this->cache_server_timestamps[i] = 0;
            this->cache_server_timestamps[i].store(0);
        }
        
        //        this->req_pq = new std::priority_queue<
        //            cache_line_t*, std::vector<cache_line_t*>, cp_comparator> ();
        
    }
    
    cacheLayer* cacheLayerThread::get_cache_layer(){
        return this->cache_layer;
    }
    
    
    double cacheLayerThread::get_minimal_timestamp(){
        return this->minimal_timestamp.load();
    }
    
    cacheLayerStat* cacheLayerThread::get_layer_stat(){
        return this->cache_layer->get_layer_stat();
    }
    
    
    
    void cacheLayerThread::run(unsigned int log_interval, const std::string log_folder){
        bool all_server_finished_local = false;
        cacheLayer *next_layer = this->cache_layer->get_next_layer();
        if (this->layer_id < NUM_CACHE_LAYERS && next_layer == NULL){
            error_msg("missing next layer pointer at layer %d\n", this->layer_id);
            abort();
        }
        
        unsigned long i;
        cache_line_t * cp;
        double min_ts;
        bool found_no_new_req;
        char akamai_stat_buf[1024];
        akamaiStat *akamai_stat = this->get_cache_layer()->akamai_stat;
        std::ofstream ofs_akamai_stat;

        std::priority_queue<cache_line_t*, std::vector<cache_line_t*>, cp_comparator> req_pq;
        
        if (log_interval != 0){
            mkdir(log_folder.c_str(), 0770);
            this->log_file_loc = std::string(log_folder) + std::string("/layer")
            + std::to_string(this->layer_id);
            this->log_filestream.open(log_file_loc);
            ofs_akamai_stat.open(std::string(log_folder)+"/akamai_stat", std::ios::out);
        }
        
        try{
            
            while (! this->all_server_finished){
                /** get all the req that have timestamp <= min_ts
                 *  order them using priorityQueue and add to servers in the layer */
                min_ts = this->get_minimal_timestamp();
                
                if (min_ts < 0.0001){
                    /* this means min_ts == 0, the initialization is not finished */
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                    debug("layer2 ready to run, no data available yet\n");
                    continue;
                }
                
                try {
                    found_no_new_req = true;
                    // this can cause one trace keeps running, while others waiting
                    for (i=0; i<this->num_of_servers; i++){
                        cp = this->cache_server_req_queue[i]->get_request(min_ts);
                        if (cp != NULL)
                            this->cache_server_req_queue[i]->queue_size_change();
                        while (cp != NULL) {
                            req_pq.push(cp);
                            found_no_new_req = false;
                            cp = this->cache_server_req_queue[i]->get_request(min_ts);
                        }
                    }
                    
//                    while (1){
//                        bool no_new_req_this_round = true;
//                        for (i=0; i<this->num_of_servers; i++){
//                            cp = this->cache_server_req_queue[i]->get_request(min_ts);
//                            if (cp != NULL){
//                                this->cache_server_req_queue[i]->queue_size_change();
//                                req_pq.push(cp);
//                                found_no_new_req = false;
//                                no_new_req_this_round = false;
//                                cp = this->cache_server_req_queue[i]->get_request(min_ts);
//                            }
//                        }
//                        if (no_new_req_this_round)
//                            break;
//                    }
                }catch (std::exception &e) {
                    std::cerr<<"err in first loop, " << e.what()<<std::endl;
                    print_stack_trace();
                    abort();
                }
                
                
                try{
                    if (req_pq.size() == 0)
                    debug("req size %lu, min ts %lf, min ts server ind %ld, min ts server ts %lu, "\
                         " its queue size %zu\n", req_pq.size(),
                          this->get_minimal_timestamp(), this->minimal_ts_server_ind.load(),
                          this->cache_server_req_queue[this->minimal_ts_server_ind.load()]->current_ts,
                          this->cache_server_req_queue[this->minimal_ts_server_ind.load()]->get_size());
#ifdef DEBUG_DEADLOCK
                    if (req_pq.size() == 0){
                        printf("req queue size ");
                        for (unsigned long i=0; i<this->num_of_servers; i++)
                            printf("%lu, ", this->cache_server_req_queue[i]->get_size());
                        printf("\n");
                        
                        printf("min ts ind ");
                        for (unsigned long i=0; i<this->num_of_servers; i++)
                            printf("%lu(%lu), ", this->cache_server_req_queue[i]->get_min_ts(),
                                   this->cache_server_req_queue[i]->get_min_ts_server_id());
                        printf("\n");
                    }
#endif 
                    while (! req_pq.empty()){
                        try{
                            cp = req_pq.top();
                            /* now add to current layer */
                            this->cache_layer->add_request(cp->cache_server_id, cp);

                            /** logic for adding missed request into next layer
                             *  currently only two layers are supported */
//                            if (!hit && this->layer_id < NUM_CACHE_LAYERS){
//                                next_layer->add_request(cp->cache_server_id, cp);
//                            }
                        }catch(std::exception &e){
                            std::cerr<<"error 1, server id " << cp->cache_server_id << " cp " << cp << e.what() << std::endl;
                            print_stack_trace();
                        }
                        
                        try{
                            /* log stat */
                            if (log_interval !=0 &&
                                cp->real_time - this->last_log_output_time > log_interval){
                                this->log_layer_stat((unsigned long) cp->real_time);
                                this->last_log_output_time = (unsigned long) cp->real_time;
                                

                                
                                
                                sprintf(akamai_stat_buf, "%lf\t%lf\t%lf\t%lf\t%lf\n",
                                        akamai_stat->get_avg_latency(),
                                        akamai_stat->get_hr_L1(),
                                        akamai_stat->get_hr_L2(),
                                        akamai_stat->get_traffic_to_origin(),
                                        akamai_stat->get_traffic_between_first_second_layer());
                                ofs_akamai_stat << std::string(akamai_stat_buf);
                                memset(akamai_stat_buf, 0, 1024);

                                
                                /* would be better to put akamaiStat output here */
                                
                            }
                        }catch(std::exception &e){
                            std::cerr<<"error 3, " << e.what() << std::endl;
                            print_stack_trace();
                        }
                        /* clean this cp and retrieve next cp */
                        req_pq.pop();
                        destroy_cacheline(cp);
                    }
                }catch (std::exception &e) {
                    std::cerr<<"err in second loop, " << e.what()<<std::endl;
                    print_stack_trace();
                }
                
                /** check whether all cache servers have finished their trace
                 *  ATTENTION: it might be better to have lock here */
                try{
                    if (found_no_new_req){
                        all_server_finished_local = true;
                        for (i=0; i<this->num_of_servers; i++){
                            if ( std::abs(LONG_MAX - (long) this->cache_server_timestamps[i].load()) > 1 ){
                                debug("server %lu not finished, ts %ld\n", i, (long) this->cache_server_timestamps[i].load());
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
                }catch (std::exception &e) {
                    std::cerr<<"err in the if, " << e.what()<<std::endl;
                    print_stack_trace();
                }
                
            }
            
            
        }catch (std::exception &e) {
            std::cerr<<e.what()<<std::endl;
            print_stack_trace();
        }
        
        
        if (log_interval != 0){
            info("layer %d finishes\n", this->layer_id); 
            this->log_filestream.close();
            ofs_akamai_stat.close();
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
     *  this function does two things 
     *  1. update cache server timestamps 
     *  2. add the requests to the layer if needed 
     *
     *  \param  cache_server_id  where the cache request comes from
     *  \param  cp               cache request
     *  \param  need_to_add      whether it is needed to add to the layer, when 
     *                           false, only update server real_ts
     */
    void cacheLayerThread::add_request(unsigned long cache_server_id,
                                       cache_line_t *cp,
                                       bool real_add){
        
        
        /** copy is done outside of mtx to reduce the time of lock
         *  if the function failed, then this memory will not be released */
        cache_line_t * cp_new = cp;
        if (real_add){
            cp_new = copy_cache_line(cp);
        }
        cp_new->cache_server_id = cache_server_id;

//        debug("server %lu add req real ts %ld, real add %d\n", cache_server_id,
//              cp->real_time, real_add);
        
        /* using lock is too expensive */
        //        std::unique_lock<std::mutex> ulock(this->mtx_add_req);
        
        

        this->cache_server_timestamps[cache_server_id].store(cp->real_time);
        
        
        /** this is not exactly mt-safe
         *  reading from cache_server_timestamps can be dangerous
         *  however, using lock here is too expensive, so prefer removing lock */
        if (this->minimal_ts_server_ind.load() == (long) cache_server_id){
            /** during this time, cache_server_timestamps on other servers may change
             *  but cache_server_timestamps on current server will not
             *  change.
             *  What this means is that,
             *  1. if the executing server thread is not the slowest thread
             *  and you find a min_ts_ind, but the min_ts_ind you find may not be
             *  the smallest at the time you retrieve its value as it may be
             *  updated during the search.
             *
             *  2. if the executing server thread is the slowest thread,
             *  then the if part will only be executed by current thread as no other thread
             *  can enter the critical section.
             *
             *  In the if part, after we load a ts (of other server),
             *  the real ts for the server might change
             *  by its server, however,
             *  even though the found min_ts may not be the smallest at the time of func return,
             *  but as long as the min_ts is smaller than true min_ts and it is making progress
             *  then it is OK */
            
            
            
            
            std::unique_lock<std::mutex> ulock(this->mtx_sync_time);
            unsigned long min_ts_ind = 0;
            double ts, min_ts = this->cache_server_timestamps[0].load();
            
            /* now need to find out the new minimal ts */
            for (unsigned long i=1; i<this->num_of_servers; i++){
                /** it is important the final value is the same as the one
                 *  when we check for minimal value */
                ts = this->cache_server_timestamps[i].load();
                /** after loading the ts, it might change during other threads */
                if (ts < min_ts){
                    min_ts = ts;
                    min_ts_ind = i;
                }
            }
            
            
            /** synchronize cache servers, this needs to be done before setting
             *  this->minimal_ts_server_ind and this->minimal_timestamp, 
             *  otherwise, deadlocks can happen 
             */
            
            /** this won't block because current thread is still the slowest thread
             *  in cache_server_req_queue */
            
            /** the places that can be blocking
             *  1. when non-slowest server is adding request to req_queue (and the queue is full). 
             *  2.
             *  the slowest server will never be blocked
             */
            this->cache_server_req_queue[cache_server_id]->add_request(cp_new, real_add);

            
            this->minimal_ts_server_ind.store(min_ts_ind);
            this->minimal_timestamp.store(min_ts);
            
            for (unsigned long i=0; i<this->num_of_servers; i++){
                this->cache_server_req_queue[i]->synchronize_time(min_ts, min_ts_ind);
            }

            ulock.unlock();
        }
        else{
        
            this->cache_server_req_queue[cache_server_id]->add_request(cp_new, real_add);
        }
    }
    
    
    
    void cacheLayerThread::cache_server_finish(unsigned long cache_server_id){
        std::stringstream ss;
        ss << "server " << cache_server_id << " finished, all server ts ";
        for (unsigned long i=0; i<this->num_of_servers; i++)
            ss << this->cache_server_timestamps[i].load() <<"\t";
        info("%s\n", ss.str().c_str());
        
        this->cache_server_timestamps[cache_server_id].store((double) (LONG_MAX));
        /** 0817 it might be possible that this server is chosen as the slowest one (in add_request)
         *  after the server checking whether it is the slowest one
         *  (this can happen when cacheServers are ending)
         *  in this case, it can deadlock, because the slowest server is gone 
         *  currently we use mtx_sync_time to ensure that no can check whether 
         *  he is the slowest when someone else is going to change it.
         */
         
         
        /** when minimal_ts_server_ind is current server, then it won't change
         *  in other threads, because only current server can change it,
         *  threrefore it is thread-safe */
        std::unique_lock<std::mutex> ulock(this->mtx_sync_time);

        if (this->minimal_ts_server_ind.load() == (long) cache_server_id){
            /* now need to find out the new minimal ts */
            unsigned long min_ts_ind = 0;
            double min_ts = this->cache_server_timestamps[0].load(), ts;
            
            /* now need to find out the new minimal ts */
            for (unsigned long i=1; i<this->num_of_servers; i++){
                ts = this->cache_server_timestamps[i].load();
                if (ts < min_ts){
                    min_ts = ts;
                    min_ts_ind = i;
                }
            }
            
            this->minimal_ts_server_ind.store(min_ts_ind);
            this->minimal_timestamp.store(min_ts);
            //            verbose("trace finish, new min ts %lf from %lu\n", min_ts, cache_server_id);
        
            // synchronize cache servers
            for (unsigned long i=0; i<this->num_of_servers; i++)
                this->cache_server_req_queue[i]->synchronize_time(min_ts, min_ts_ind);
        }
        ulock.unlock();
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
