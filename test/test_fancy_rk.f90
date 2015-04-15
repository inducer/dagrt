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

  type(sim_grid_state_type), pointer, dimension(:) :: initial_condition

  type(leap_state_type), target :: leap_state
  type(leap_state_type), pointer :: leap_state_ptr

  real*8 t_fin
  integer ntrips, igrid, icells, idofs
  parameter (t_fin=1d0, ntrips=20)

  integer istep

  ! start code ----------------------------------------------------------------

  leap_state_ptr => leap_state
  region_ptr => region
 
  region%nGrids = 2

  allocate(region%nCells(region%nGrids))
  allocate(region%n_grid_dofs(region%nGrids))

  do igrid = 1, region%nGrids
    region%nCells(igrid) = 2
    region%n_grid_dofs(igrid) = 2
    allocate(initial_condition(igrid)%conserved_var(region%n_grid_dofs(igrid),region%nCells(igrid)))
    allocate(initial_condition(igrid)%rhs(region%n_grid_dofs(igrid),region%nCells(igrid)))
    do icells=1, region%nCells(igrid)
      do idofs = 1, region%n_grid_dofs(igrid)
        initial_condition(igrid)%conserved_var(idofs,icells) = 1
        initial_condition(igrid)%rhs(idofs,icells) = 0
        end do
    end do
  end do

  call timestep_initialize( &
    region=region_ptr, &
    leap_state=leap_state_ptr, &
    state_y=initial_condition, &
    leap_t=0d0, &
    leap_dt=t_fin/20)

  do istep = 1,ntrips
    call timestep_run(region=region_ptr, leap_state=leap_state_ptr)
  enddo

  call timestep_shutdown(region=region_ptr, leap_state=leap_state_ptr)

end program

