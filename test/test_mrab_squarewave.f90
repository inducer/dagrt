program test_mrabmethod_squarewave
  use MRAB, only: leap_state_type, &
    timestep_initialize => initialize, &
    timestep_run => run, &
    timestep_shutdown => shutdown, &
    leap_state_func_initialization, &
    leap_state_func_primary

  implicit none

  type(leap_state_type), target :: state
  type(leap_state_type), pointer :: state_ptr

  real*8, dimension(2) :: initial_condition 
  real*8, dimension(1) :: true_sol_fast, true_sol_slow
  integer, dimension(2) :: ntrips

  integer run_count, k
  real*8 t_fin
  parameter (run_count=2, t_fin=1d0)

  real*8, dimension(run_count):: dt_values, error_slow, error_fast

  real*8 min_order, est_order_fast, est_order_slow

  integer stderr
  parameter(stderr=0)

  integer irun

  ! start code ----------------------------------------------------------------

  state_ptr => state

  initial_condition(1) = (exp(0d0))*cos(0d0) ! fast
  initial_condition(2) = (exp(0d0))*sin(0d0) ! slow

  ntrips(1) = NUM_TRIPS_ONE
  ntrips(2) = NUM_TRIPS_TWO

  do irun = 1,run_count
    dt_values(irun) = t_fin/ntrips(irun)

    call timestep_initialize( &
      leap_state=state_ptr, &
      state_slow=initial_condition(2:2), &
      state_fast=initial_condition(1:1), &
      leap_t=0d0, &
      leap_dt=dt_values(irun))

    k = 0

    do
      if (k == 1) then
        state%leap_dt = dt_values(irun)/4
        k = 0
      else
        state%leap_dt = dt_values(irun)
        k = 1
      endif
      call timestep_run(leap_state=state_ptr)
      if (state%ret_time_fast.ge.t_fin) then
        exit
      endif
    end do 

    true_sol_fast = (exp(state%ret_time_fast))*cos(state%ret_time_fast)
    true_sol_slow = (exp(state%ret_time_slow))*sin(state%ret_time_slow)

    error_slow(irun) = sqrt(sum((true_sol_slow-state%ret_state_slow)**2))
    error_fast(irun) = sqrt(sum((true_sol_fast-state%ret_state_fast)**2))

    call timestep_shutdown(leap_state=state_ptr)
    write(*,*) 'done', dt_values(irun), error_slow(irun), error_fast(irun)
  enddo

  min_order = MIN_ORDER
  est_order_slow = log(error_slow(2)/error_slow(1))/log(dt_values(2)/dt_values(1))
  est_order_fast = log(error_fast(2)/error_fast(1))/log(dt_values(2)/dt_values(1))

  write(*,*) 'estimated order slow:', est_order_slow
  if (est_order_slow < min_order) then
    write(stderr,*) 'ERROR: achieved order too low:', est_order_slow, ' < ', &
        min_order
  endif

  write(*,*) 'estimated order fast:', est_order_fast
  if (est_order_fast < min_order) then
    write(stderr,*) 'ERROR: achieved order too low:', est_order_fast, ' < ', &
        min_order
  endif

end program
