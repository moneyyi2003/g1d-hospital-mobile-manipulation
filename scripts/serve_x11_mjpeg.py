#!/usr/bin/env python3
"""Serve the Isaac X11 desktop as a low-latency multipart JPEG stream."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess


PAGE = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Isaac Sim H.264 View</title>
<style>
html,body{width:100%;height:100%;margin:0;background:#050607;overflow:hidden}
video{width:100%;height:100%;object-fit:contain;display:block}
.tag{position:fixed;left:10px;top:10px;padding:6px 9px;border-radius:6px;
background:#000a;color:#dfe8ef;font:13px system-ui;pointer-events:none}
</style><video id="view" autoplay muted playsinline></video>
<div class="tag">Isaac Sim · H.264 · 30 FPS</div>
<script type="module">
const video=document.querySelector('#view');
const media=new MediaSource(); video.src=URL.createObjectURL(media);
media.addEventListener('sourceopen',async()=>{
  const mime='video/mp4; codecs="avc1.42c020"';
  if(!MediaSource.isTypeSupported(mime)){document.querySelector('.tag').textContent='浏览器不支持 H.264 MSE';return}
  const buffer=media.addSourceBuffer(mime), queue=[];
  const pump=()=>{
    if(buffer.updating||!queue.length)return;
    buffer.appendBuffer(queue.shift());
  };
  buffer.addEventListener('updateend',()=>{
    if(buffer.buffered.length){
      const end=buffer.buffered.end(buffer.buffered.length-1);
      if(end-video.currentTime>0.8) video.currentTime=Math.max(0,end-0.15);
      if(end>12&&!buffer.updating) try{buffer.remove(0,end-8)}catch(_){}
    }
    pump();
  });
  const response=await fetch('/stream.mp4',{cache:'no-store'});
  const reader=response.body.getReader();
  while(true){const {done,value}=await reader.read();if(done)break;queue.push(value);pump()}
});
</script>
</html>""".encode("utf-8")

MJPEG_PAGE = """<!doctype html><html><meta charset="utf-8">
<style>html,body{width:100%;height:100%;margin:0;background:#050607}img{width:100%;height:100%;object-fit:contain}</style>
<img src="/stream.mjpg"></html>""".encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "IsaacMJPEG/1.0"

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html", "/h264.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
            return
        if path == "/mjpeg.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(MJPEG_PAGE)))
            self.end_headers()
            self.wfile.write(MJPEG_PAGE)
            return
        if path == "/stream.mp4":
            self._serve_h264()
            return
        if path != "/stream.mjpg":
            self.send_error(404)
            return
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-f", "x11grab", "-draw_mouse", "0",
            "-framerate", str(self.server.fps),
            "-video_size", self.server.capture_size,
            "-i", self.server.display,
            "-vf", f"scale={self.server.width}:-2",
            "-an", "-c:v", "mjpeg", "-q:v", str(self.server.quality),
            "-f", "mpjpeg", "pipe:1",
        ]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace;boundary=ffmpeg"
        )
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        try:
            assert process.stdout is not None
            while chunk := process.stdout.read(65536):
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

    def _serve_h264(self) -> None:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-f", "x11grab", "-draw_mouse", "0", "-framerate", "30",
            "-video_size", self.server.capture_size, "-i", self.server.display,
            "-vf", "scale=1600:750", "-an", "-c:v", "libx264",
            "-preset", "ultrafast", "-tune", "zerolatency",
            "-profile:v", "baseline", "-pix_fmt", "yuv420p",
            "-b:v", "5M", "-maxrate", "5M", "-bufsize", "1M",
            "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
            "-movflags", "frag_keyframe+empty_moov+default_base_moof",
            "-frag_duration", "100000", "-f", "mp4", "pipe:1",
        ]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            assert process.stdout is not None
            while chunk := process.stdout.read(65536):
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

    def log_message(self, _format: str, *_args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6082)
    parser.add_argument("--display", default=":1.0")
    parser.add_argument("--capture-size", default="1920x900")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=6)
    options = parser.parse_args()
    server = ThreadingHTTPServer((options.host, options.port), Handler)
    server.display = options.display
    server.capture_size = options.capture_size
    server.width = options.width
    server.fps = options.fps
    server.quality = options.quality
    server.serve_forever()


if __name__ == "__main__":
    main()
