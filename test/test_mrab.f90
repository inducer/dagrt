program test_mrabmethod
  use MRAB, only: leap_state_type, &
    timestep_initialize => initialize, &
    timestep_run => run, &
    timestep_shutdown => shutdown, &
    leap_state_func_initialization, &
    leap_state_func_primary

  implicit none

  type(leap_state_type), target :: state
  type(leap_state_type), pointer :: state_ptr

  real*8, dimension(2) :: initial_condition, true_sol_fast, true_sol_slow
  integer, dimension(2) :: ntrips

  integer run_count
  real*8 t_fin
  parameter (run_count=2, t_fin=1d0)

  real*8, dimension(run_count):: dt_values, error_slow, error_fast

  real*8 est_order, min_order, est_order_fast, est_order_slow

  integer stderr
  parameter(stderr=0)

  integer istep, irun

  ! start code ----------------------------------------------------------------

  state_ptr => state

  initial_condition(1) = 2
  initial_condition(2) = 2.3

  ntrips(1) = 20
  ntrips(2) = 50

  do irun = 1,run_count
    dt_values(irun) = t_fin/ntrips(irun)

    call timestep_initialize( &
      leap_state=state_ptr, &
      state_slow=initial_condition, &
      state_fast=initial_condition, &
      leap_t=0d0, &
      leap_dt=dt_values(irun))

    do istep = 1,ntrips(irun)
      call timestep_run(leap_state=state_ptr)
      write(*,*) state%ret_state_fast, state%ret_state_slow
    enddo

    true_sol_fast = initial_condition * (exp(2*state%ret_time_fast))
    true_sol_slow = initial_condition * (exp(state%ret_time_slow))

    error_slow(irun) = sqrt(sum((true_sol_slow-state%ret_state_slow)**2))
    error_fast(irun) = sqrt(sum((true_sol_fast-state%ret_state_fast)**2))

    write(*,*) error_fast

    call timestep_shutdown(leap_state=state_ptr)
    write(*,*) 'done'
  enddo

  min_order = MIN_ORDER
  est_order_slow = log(error_slow(2)/error_slow(1))/log(dt_values(2)/dt_values(1))
  est_order_fast = log(error_fast(2)/error_fast(1))/log(dt_values(2)/dt_values(1))

  write(*,*) 'estimated order slow:', est_order_slow
  if (est_order_slow < min_order) then
    write(stderr,*) 'ERROR: achieved order too low:', est_order, ' < ', &
        min_order
  endif

  write(*,*) 'estimated order fast:', est_order_fast
  if (est_order_fast < min_order) then
    write(stderr,*) 'ERROR: achieved order too low:', est_order, ' < ', &
        min_order
  endif

end program
