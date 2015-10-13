program test_rkmethod

  use arrays, only: dagrt_state_type, &
    timestep_initialize => initialize, &
    timestep_run => run, &
    timestep_shutdown => shutdown

  implicit none

  type(dagrt_state_type), target :: dagrt_state
  type(dagrt_state_type), pointer :: dagrt_state_ptr

  real*8 t_fin
  integer ntrips
  parameter (t_fin=1d0, ntrips=20)

  ! start code ----------------------------------------------------------------

  dagrt_state_ptr => dagrt_state

  call timestep_initialize( &
    dagrt_state=dagrt_state_ptr, &
    dagrt_t=0d0, &
    dagrt_dt=t_fin/20)

  call timestep_run(dagrt_state=dagrt_state_ptr)

  call timestep_shutdown(dagrt_state=dagrt_state_ptr)

end program

