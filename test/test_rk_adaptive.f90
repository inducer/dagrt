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

  real*8, dimension(2) :: initial_condition
  real*8, dimension(100) :: step_sizes

  real*8 t_fin, dt_value, small_step_frac, big_step_frac, old_time
  parameter (t_fin=1d0)

  integer stderr
  parameter(stderr=0)

  integer istep, nruns, num_big_steps, num_small_steps
  parameter (nruns = 100)

  ! start code ----------------------------------------------------------------

  state_ptr => state

  initial_condition(1) = 2
  initial_condition(2) = 0

  dt_value = 1e-5
  old_time = 0d0
  num_big_steps = 0
  num_small_steps = 0

  call timestep_initialize( &
      leap_state=state_ptr, &
      state_y=initial_condition, &
      leap_t=0d0, &
      leap_dt=dt_value)

  do istep = 1, nruns

    call timestep_run(leap_state=state_ptr)

    step_sizes(istep) = state%ret_time_y - old_time
    old_time = state%ret_time_y

    if (step_sizes(istep)<0.01d0) then
      num_small_steps = num_small_steps + 1
    elseif (step_sizes(istep)>0.05d0) then
      num_big_steps = num_big_steps + 1
    endif

  enddo

  call timestep_shutdown(leap_state=state_ptr)
  write(*,*) 'done'

  big_step_frac = num_big_steps/real(nruns)
  small_step_frac = num_small_steps/real(nruns)

  if (big_step_frac>=0.16d0 .and. small_step_frac<=0.35d0) then
    write(*,*), "Test passes: big_step_frac = ", big_step_frac
    write(*,*), "Test passes: small_step_frac = ", small_step_frac
  else
    write(*,*), "Test fails: big_step_frac = ", big_step_frac
    write(*,*), "Test fails: small_step_frac = ", small_step_frac
  endif

end program

