def run():
    import sys
    from couchdb.client import Server as CouchServer
    csrv = CouchServer(sys.argv[1])
    cdb = csrv[sys.argv[2]]

    from datapyle.couch_queue import populate_queue
    from mrab_stability import generate_mrab_jobs, generate_mrab_jobs_hires
    #populate_queue(generate_mrab_jobs, cdb)
    populate_queue(generate_mrab_jobs_hires, cdb)

    #enter_queue_manager(generate_srab_jobs, "output-srab.dat")
    #enter_queue_manager(generate_mrab_jobs_hires, "output-hires.dat")
    #enter_queue_manager(generate_mrab_jobs_step_verify, "output-step.dat")




if __name__ == "__main__":
    run()
