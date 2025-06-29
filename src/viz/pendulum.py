import jax.numpy as jnp
from jaxtyping import Array, Float
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import numpy as np
from typing import List, Optional


class PendulumVisualizer:
    def __init__(
        self,
        length: float = 1.0,
        figsize: tuple = (8, 8),
        trail_length: int = 20,
        show_angle_arc: bool = True,
        show_velocity_arrow: bool = True,
        dark_mode: bool = False,
    ):
        self.length = length
        self.figsize = figsize
        self.trail_length = trail_length
        self.show_angle_arc = show_angle_arc
        self.show_velocity_arrow = show_velocity_arrow
        self.dark_mode = dark_mode

        # Color scheme
        if dark_mode:
            self.bg_color = "#1a1a1a"
            self.grid_color = "#333333"
            self.text_color = "#ffffff"
            self.rod_color = "#cccccc"
            self.bob_color = "#ff6b6b"
            self.trail_color = "#4ecdc4"
            self.arc_color = "#ffe66d"
        else:
            self.bg_color = "#ffffff"
            self.grid_color = "#cccccc"
            self.text_color = "#333333"
            self.rod_color = "#333333"
            self.bob_color = "#e74c3c"
            self.trail_color = "#3498db"
            self.arc_color = "#f39c12"

    def create_animation(
        self,
        states: List[Float[Array, "2"]],
        actions: Optional[List[Float[Array, "1"]]] = None,
        rewards: Optional[List[Float[Array, ""]]] = None,
        filename: str = "pendulum.mp4",
        fps: int = 30,
        bitrate: int = 2400,
    ):
        fig, ax = plt.subplots(figsize=self.figsize)
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.bg_color)

        # Setup axes
        lim = self.length * 1.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, color=self.grid_color)
        ax.set_xlabel("X", color=self.text_color)
        ax.set_ylabel("Y", color=self.text_color)
        ax.tick_params(colors=self.text_color)

        # Pendulum components
        (rod_line,) = ax.plot(
            [],
            [],
            "o-",
            lw=4,
            color=self.rod_color,
            markersize=8,
            markerfacecolor=self.grid_color,
        )
        bob = Circle((0, 0), 0.08 * self.length, color=self.bob_color, zorder=5)
        ax.add_patch(bob)

        # Trail for bob position
        trail_x, trail_y = [], []
        (trail_line,) = ax.plot(
            [], [], "-", color=self.trail_color, alpha=0.5, lw=2, zorder=1
        )

        # Angle arc
        if self.show_angle_arc:
            from matplotlib.patches import Arc

            angle_arc = Arc(
                (0, 0),
                0.3 * self.length,
                0.3 * self.length,
                angle=0,
                theta1=-90,
                theta2=0,
                color=self.arc_color,
                lw=2,
            )
            ax.add_patch(angle_arc)

        # Velocity arrow
        if self.show_velocity_arrow:
            velocity_arrow = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=2),
            )

        # Info text box
        info_box = FancyBboxPatch(
            (lim * 0.4, lim * 0.7),
            lim * 0.55,
            lim * 0.25,
            boxstyle="round,pad=0.05",
            facecolor=self.bg_color,
            edgecolor=self.grid_color,
            alpha=0.8,
        )
        ax.add_patch(info_box)

        info_text = ax.text(
            lim * 0.45,
            lim * 0.85,
            "",
            fontsize=10,
            color=self.text_color,
            verticalalignment="top",
        )

        # Torque indicator
        if actions is not None:
            torque_bar = Rectangle(
                (-lim * 0.8, -lim * 0.9),
                0,
                0.1 * lim,
                facecolor="#9b59b6",
                edgecolor=self.text_color,
            )
            ax.add_patch(torque_bar)
            ax.text(
                -lim * 0.8,
                -lim * 0.75,
                "Torque",
                fontsize=9,
                color=self.text_color,
                ha="center",
            )

        def init():
            rod_line.set_data([], [])
            trail_line.set_data([], [])
            trail_x.clear()
            trail_y.clear()
            return [rod_line, bob, trail_line, info_text]

        def animate_frame(i):
            if i >= len(states):
                return [rod_line, bob, trail_line, info_text]

            theta, theta_dot = states[i]

            # Pendulum position
            x = float(self.length * jnp.sin(theta))
            y = float(-self.length * jnp.cos(theta))

            # Update rod
            rod_line.set_data([0, x], [0, y])
            bob.center = (x, y)

            # Update trail
            trail_x.append(x)
            trail_y.append(y)
            if len(trail_x) > self.trail_length:
                trail_x.pop(0)
                trail_y.pop(0)
            trail_line.set_data(trail_x, trail_y)

            # Update angle arc
            if self.show_angle_arc:
                angle_deg = float(jnp.degrees(theta))
                angle_arc.theta2 = angle_deg - 90

            # Update velocity arrow
            if self.show_velocity_arrow:
                vel_scale = 0.2 * self.length
                vel_x = vel_scale * theta_dot * jnp.cos(theta)
                vel_y = vel_scale * theta_dot * jnp.sin(theta)
                velocity_arrow.set_position((x, y))
                velocity_arrow.xy = (x + vel_x, y + vel_y)

            # Update info text
            info_lines = [
                f"Step: {i:4d}",
                f"θ: {float(jnp.degrees(theta)):6.1f}°",
                f"ω: {float(theta_dot):6.2f} rad/s",
            ]

            if rewards is not None and i > 0:
                info_lines.append(f"R: {float(rewards[i - 1]):6.3f}")

            info_text.set_text("\n".join(info_lines))

            # Update torque indicator
            if actions is not None and i > 0:
                torque = float(actions[i - 1][0])
                max_torque = 2.0  # Should match environment
                torque_width = (torque / max_torque) * lim * 0.3
                torque_bar.set_width(torque_width)
                torque_bar.set_x(
                    -lim * 0.8 if torque >= 0 else -lim * 0.8 + torque_width
                )

            return [rod_line, bob, trail_line, info_text]

        anim = animation.FuncAnimation(
            fig,
            animate_frame,
            init_func=init,
            frames=len(states),
            interval=1000 / fps,
            blit=False,
            repeat=True,
        )

        # Save animation
        writer = animation.FFMpegWriter(
            fps=fps, metadata=dict(artist="JAX-RL Pendulum"), bitrate=bitrate
        )
        anim.save(filename, writer=writer)
        plt.close(fig)

        return filename

    def create_phase_portrait(
        self,
        states: List[Float[Array, "2"]],
        filename: str = "phase_portrait.png",
        show_trajectory: bool = True,
        show_vector_field: bool = True,
    ):
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.bg_color)

        # Extract trajectory
        thetas = np.array([float(s[0]) for s in states])
        theta_dots = np.array([float(s[1]) for s in states])

        # Vector field
        if show_vector_field:
            theta_range = np.linspace(-np.pi, np.pi, 20)
            theta_dot_range = np.linspace(-8, 8, 15)
            Theta, ThetaDot = np.meshgrid(theta_range, theta_dot_range)

            # Pendulum dynamics without control
            g, L = 10.0, 1.0
            dTheta = ThetaDot
            dThetaDot = -(g / L) * np.sin(Theta)

            # Normalize for visualization
            magnitude = np.sqrt(dTheta**2 + dThetaDot**2)
            dTheta_norm = dTheta / (magnitude + 1e-6)
            dThetaDot_norm = dThetaDot / (magnitude + 1e-6)

            ax.quiver(
                Theta,
                ThetaDot,
                dTheta_norm,
                dThetaDot_norm,
                magnitude,
                cmap="viridis",
                alpha=0.6,
                scale=30,
            )

        # Trajectory
        if show_trajectory:
            ax.plot(
                thetas,
                theta_dots,
                color=self.trail_color,
                lw=2,
                alpha=0.8,
                label="Trajectory",
            )
            ax.scatter(
                thetas[0],
                theta_dots[0],
                color="green",
                s=100,
                marker="o",
                label="Start",
                zorder=5,
            )
            ax.scatter(
                thetas[-1],
                theta_dots[-1],
                color="red",
                s=100,
                marker="s",
                label="End",
                zorder=5,
            )

        ax.set_xlabel("Angle θ (rad)", fontsize=12, color=self.text_color)
        ax.set_ylabel("Angular velocity ω (rad/s)", fontsize=12, color=self.text_color)
        ax.set_title("Pendulum Phase Portrait", fontsize=14, color=self.text_color)
        ax.grid(True, alpha=0.3, color=self.grid_color)
        ax.tick_params(colors=self.text_color)
        ax.legend(
            loc="upper right",
            facecolor=self.bg_color,
            edgecolor=self.grid_color,
            labelcolor=self.text_color,
        )

        plt.tight_layout()
        plt.savefig(filename, dpi=150, facecolor=self.bg_color)
        plt.close(fig)

        return filename
