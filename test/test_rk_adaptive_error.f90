program test_rkmethod
  use RKMethod, only: leap_state_type, &
    timestep_initialize => initialize, &
    timestep_run => run, &
    timestep_shutdown => shutdown, &
    leap_state_func_initialization, &
    leap_state_func_primary

  implicit none

  type(leap_state_type), target :: state
  type(leap_state_type), pointer :: state_ptr

  real*8, dimension(2) :: initial_condition, true_sol

  real*8 t_fin
  integer ntrips
  parameter (t_fin=1d0, ntrips=100)

  real*8 :: dt_value, error, atol

  integer stderr
  parameter(stderr=0)

  integer istep

  ! start code ----------------------------------------------------------------

  state_ptr => state

  initial_condition(1) = 2
  initial_condition(2) = 2.3

  dt_value = 1e-5
  atol = 1e-6

  call timestep_initialize( &
    leap_state=state_ptr, &
    state_y=initial_condition, &
    leap_t=0d0, &
    leap_dt=dt_value)

  do istep = 1, ntrips
    call timestep_run(leap_state=state_ptr)
  enddo

  true_sol = initial_condition * exp(-2*state%ret_time_y)
  error = sqrt(sum((true_sol-state%ret_state_y)**2))

  call timestep_shutdown(leap_state=state_ptr)
  write(*,*) 'done'

  if (error>atol) then
    write(stderr,*) 'ERROR: Max error exceeds tolerance:', error, ' < ', &
        atol
  endif

end program
