program test_rkmethod

  use RKMethod, only: leap_state_type, &
    timestep_initialize => initialize, &
    timestep_run => run, &
    timestep_shutdown => shutdown, &
    leap_state_func_initialization, &
    leap_state_func_primary

  use sim_types

  implicit none

  type(region_type), target :: region
  type(region_type), pointer :: region_ptr

  type(leap_state_type), target :: state
  type(leap_state_type), pointer :: state_ptr

  real*8, dimension(2) :: initial_condition

  real*8 t_fin
  integer ntrips
  parameter (t_fin=1d0, ntrips=20)

  integer istep

  ! start code ----------------------------------------------------------------

  state_ptr => state
  region_ptr => region

  call timestep_initialize( &
    region=region_ptr, &
    leap_state=state_ptr, &
    state_y=initial_condition, &
    leap_t=0d0, &
    leap_dt=t_fin/20)

  do istep = 1,ntrips
    call timestep_run(region=region_ptr, leap_state=state_ptr)
  enddo

  call timestep_shutdown(region=region_ptr, leap_state=state_ptr)
end program

