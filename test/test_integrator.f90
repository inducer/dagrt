program test_rkmethod
  use RKMethod, only: leap_state_type, &
    timestep_initialize => initialize, &
    timestep_shutdown => shutdown, &
    leap_state_func_initialization, &
    leap_state_func_primary

  implicit none

  type(leap_state_type), target :: state
  type(leap_state_type), pointer :: state_ptr

  real*8, dimension(2) :: initial_condition, true_sol
  integer, dimension(2) :: nsteps

  integer run_count
  real*8 t_fin
  parameter (run_count=2, t_fin=1d0)

  real*8, dimension(run_count):: dt_values, errors

  integer istep, irun

  ! start code ----------------------------------------------------------------

  state_ptr => state

  initial_condition(1) = 2
  initial_condition(2) = 2.3
  true_sol = initial_condition * exp(-2*t_fin)

  nsteps(1) = 2000
  nsteps(2) = 5000

  do irun = 1,run_count
    dt_values(irun) = t_fin/nsteps(irun)

    call timestep_initialize( &
      leap_state=state_ptr, &
      state_y=initial_condition, &
      leap_t=0d0, &
      leap_dt=dt_values(irun))

    call leap_state_func_initialization(leap_state=state_ptr)
    do istep = 1,nsteps(irun)
      call leap_state_func_primary(leap_state=state_ptr)
    enddo

    errors(irun) = sqrt(sum((true_sol-state%ret_state_y)**2))

    call timestep_shutdown(leap_state=state_ptr)
    write(*,*) 'done'
  enddo
  write(*,*) 'estimated order:', &
    log(errors(2)/errors(1))/log(dt_values(2)/dt_values(1))

end program
