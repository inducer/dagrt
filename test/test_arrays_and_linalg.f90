program test_rkmethod

  use arrays, only: leap_state_type, &
    timestep_initialize => initialize, &
    timestep_run => run, &
    timestep_shutdown => shutdown

  implicit none

  type(leap_state_type), target :: leap_state
  type(leap_state_type), pointer :: leap_state_ptr

  real*8 t_fin
  integer ntrips, igrid, idof
  parameter (t_fin=1d0, ntrips=20)

  ! start code ----------------------------------------------------------------

  leap_state_ptr => leap_state

  call timestep_initialize( &
    leap_state=leap_state_ptr, &
    leap_t=0d0, &
    leap_dt=t_fin/20)

  call timestep_run(leap_state=leap_state_ptr)

  call timestep_shutdown(leap_state=leap_state_ptr)

end program

