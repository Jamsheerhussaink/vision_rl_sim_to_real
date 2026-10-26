"""Standalone ImGui reward charts; receives read-only samples over stdin."""
import json
import os


def run():
    from imgui_bundle import imgui, immapp, hello_imgui
    from isaac_vision_rl.utils.reward_panel import RewardPanel
    panel=RewardPanel([])
    panel.chart_note='LIVE PPO TRAINING | Environment 0 only | Read-only reward charts'
    os.set_blocking(0,False)
    pending=b''

    def draw():
        nonlocal pending
        # Bound work per GUI frame; an unavailable display never blocks PPO.
        for _ in range(16):
            try:data=os.read(0,65536)
            except BlockingIOError:break
            if not data:break
            pending+=data
        lines=pending.split(b'\n');pending=lines.pop()
        for line in lines:
            sample=json.loads(line)
            panel.snapshot=sample
            panel.record(sample)
        if panel.snapshot:
            if getattr(panel, "sample_count", 0) % 15 == 0:
                try:
                    snap_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../logs/live_observatory.json"))
                    with open(snap_path, "w") as f:
                        json.dump(panel.snapshot, f)
                except Exception:
                    pass
            panel.draw_charts(imgui,standalone=True)
        else:
            imgui.begin('Reward charts')
            imgui.text('Waiting for training reward samples...')
            imgui.end()

    params=hello_imgui.RunnerParams()
    params.app_window_params.window_title='SO-101 Reward Observatory'
    params.app_window_params.window_geometry.size=(900,960)
    params.app_window_params.restore_previous_geometry=False
    params.app_window_params.borderless=True
    params.app_window_params.borderless_movable=True
    params.app_window_params.borderless_resizable=True
    params.imgui_window_params.default_imgui_window_type=hello_imgui.DefaultImGuiWindowType.no_default_window
    params.imgui_window_params.show_menu_bar=False
    params.imgui_window_params.show_status_bar=False
    params.callbacks.show_gui=draw

    def style():
        imgui.style_colors_dark()
        theme=imgui.get_style()
        theme.window_padding=imgui.ImVec2(24,20)
        theme.item_spacing=imgui.ImVec2(12,10)
        theme.frame_padding=imgui.ImVec2(8,5)
        theme.window_rounding=10
        theme.frame_rounding=5
        theme.grab_rounding=5
        theme.window_border_size=0
        for key,color in (
            (imgui.Col_.window_bg,(.035,.047,.067,1.)),
            (imgui.Col_.frame_bg,(.075,.10,.14,1.)),
            (imgui.Col_.frame_bg_hovered,(.10,.17,.22,1.)),
            (imgui.Col_.check_mark,(.25,.86,.92,1.)),
            (imgui.Col_.slider_grab,(.25,.76,.85,1.)),
            (imgui.Col_.separator,(.13,.20,.27,1.)),
            (imgui.Col_.text,(.86,.90,.95,1.))):
            theme.set_color_(key,imgui.ImVec4(*color))
    params.callbacks.setup_imgui_style=style
    addons=immapp.AddOnsParams()
    addons.with_implot=True
    immapp.run(params,addons)


class RewardWindow:
    """Nonblocking sender: closing/slowing the charts cannot stall training."""
    def __init__(self):
        import atexit
        import subprocess
        import sys
        self.process=subprocess.Popen([sys.executable,'-m',__name__],stdin=subprocess.PIPE)
        os.set_blocking(self.process.stdin.fileno(),False)
        atexit.register(self.close)

    def send(self,sample):
        if self.process.poll() is not None:return
        payload=(json.dumps(sample)+'\n').encode()
        # Atomic writes prevent a full pipe from producing partial JSON records.
        if len(payload)>4096:return
        try:os.write(self.process.stdin.fileno(),payload)
        except (BlockingIOError,BrokenPipeError):pass

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=2)
            except Exception:self.process.kill()


if __name__=='__main__':run()
