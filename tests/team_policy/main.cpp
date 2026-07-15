#include <Kokkos_Core.hpp>

#include <kkf/extractor.hpp>

// Kokkos::TeamHandle is only available from Kokkos 5.1.0
template <class T>
concept TeamHandle = Kokkos::is_team_handle_v<T>;

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values("values", N);
  Kokkos::parallel_for(
      "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  cexa::kernel_replayer::parallel_for(
      "test_kernel", Kokkos::TeamPolicy(3, Kokkos::AUTO),
      KOKKOS_LAMBDA(TeamHandle auto team) {
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, 1024), [&](int i) {
          values(i) *= team.team_size() + team.league_size();
        });
      });
  Kokkos::fence();

  auto h_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  Kokkos::printf("values(5) = %d\n", h_values(5));

  return 0;
}
