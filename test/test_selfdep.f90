program test_selfdep

  use selfdep, only: dagrt_state_type, &
    timestep_initialize => initialize, &
    timestep_run => run, &
    timestep_shutdown => shutdown

  implicit none

  type(dagrt_state_type), target :: dagrt_state
  type(dagrt_state_type), pointer :: dagrt_state_ptr

  real*8, dimension(100) :: y0

  integer i

  ! start code ----------------------------------------------------------------

  dagrt_state_ptr => dagrt_state


  do i = 1, 100
    y0 = i
  end do

  call timestep_initialize(dagrt_state=dagrt_state_ptr, state_y=y0)
  call timestep_run(dagrt_state=dagrt_state_ptr)
  call timestep_shutdown(dagrt_state=dagrt_state_ptr)

end program

