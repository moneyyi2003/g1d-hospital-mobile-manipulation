# Isaac Sim 6.0.1 WebRTC Streaming 排查记录

更新时间：2026-07-23（UTC）

## 当前诊断结论

Isaac Sim、RTX 渲染、NVENC 库和 WebRTC extensions 均正常加载。浏览器停在
`WAITING FOR STREAM` 的根因是当前 AutoDL 公有云实例没有可从浏览器访问的 UDP 媒体
路径，不是 GPU 驱动、Isaac extension 或缺少 `streamPort` 配置。

Isaac Sim WebRTC 需要两条独立路径：

- `49100/TCP`：连接协商和信令；
- `47998/UDP`：SRTP 视频、输入和 WebRTC 媒体。

AutoDL 当前容器只有 `172.17.0.14` 私网地址，没有独立公网 IP。AutoDL 自定义服务和
SSH 隧道可以代理 TCP/HTTP，但不能把浏览器所需的 `47998/UDP` 直接转发到此容器。
因此页面和 TCP 信令可达并不代表视频链路可达。

## 已核对的本机证据

当前 Kit 日志：

```text
/root/.nvidia-omniverse/logs/Kit/Isaac-Sim Streaming/6.0/kit_20260723_142125.log
```

日志确认：

```text
omni.kit.livestream.app 10.1.1
omni.kit.livestream.webrtc 10.3.2
omni.kit.livestream.core 10.2.1
Started primary stream server on signal port 49100 and stream port 47998
```

当前配置位于：

```text
/root/autodl-tmp/isaacsim/apps/isaacsim.exp.full.streaming.kit
```

其中已经设置：

```toml
primaryStream.signalPort = 49100
primaryStream.streamPort = 47998
primaryStream.streamType = "webrtc"
```

其他检查结果：

- Kit 持续渲染 viewport，RTX 初始化成功。
- `libnvidia-encode.so` 和 `libnvcuvid.so` 已由驱动提供。
- 49100/TCP 正常监听。
- 47998/UDP 没有形成持续可见的媒体 socket。
- 多个 `NvStreamer-*.etli` 每隔约 15–18 秒生成，但只有 session 初始化字段，没有成功
  ICE candidate pair 或视频统计；这与客户端反复协商、媒体路径失败相符。
- 当前 Web 客户端曾把页面 hostname 同时作为 `mediaServer`。当页面通过 AutoDL
  HTTP/TCP 代理或 SSH TCP 隧道访问时，该 hostname 并不是 47998/UDP 的可达终点。
- 容器没有 `/dev/net/tun` 和 `CAP_NET_ADMIN`，不能在容器内直接建立内核级 VPN/TUN
  来补出 UDP 路径。

## 47998 UDP 为什么看起来没有监听

`streamPort=47998` 已经传给 StreamSDK，但固定端口配置不等于当前一定存在长期 UDP
socket。WebRTC 媒体 transport 会在客户端会话和 ICE 协商阶段建立。当前连接在有效
candidate pair 形成前反复超时，因此检查空闲时看不到 47998/UDP，ETLI 也没有实际媒体
统计。

即使强制让进程预先 bind 47998，也不能解决浏览器无法通过 AutoDL TCP/HTTP 代理发送
UDP 包的问题。

## 可工作的部署条件

必须先满足以下方案之一：

1. 给实例/宿主机提供公网 IP，并将 `49100/TCP` 和 `47998/UDP` 都映射到容器；
2. 让服务器与浏览器设备加入同一个支持 UDP 的覆盖网络，并使用服务器的覆盖网络 IP；
3. 提供外部 TURN relay 及凭据，并确认 Isaac Sim 6.0.1 StreamSDK/客户端使用该 relay。

当前 AutoDL 公有云 TCP/HTTP 自定义服务本身不满足以上任一条件。单独增加
`publicIp` 参数、Nginx、WebSocket、SSH `-L` 或改网页端口都无法替代 UDP 媒体路径。

## 获得 UDP 可达地址后的启动命令

先停止旧进程，再用同一个对外可达 IP 启动。`<UDP_REACHABLE_IP>` 必须是浏览器能通过
47998/UDP 到达的地址，不能填写 AutoDL HTTP 代理域名或 `curl ifconfig.me` 返回的共享
出口代理地址。

```bash
cd /root/autodl-tmp
export OMNI_KIT_ALLOW_ROOT=1
export OMNI_KIT_ACCEPT_EULA=YES

./isaacsim/isaac-sim.streaming.sh \
  --/exts/omni.kit.livestream.app/primaryStream/publicIp=<UDP_REACHABLE_IP> \
  --/exts/omni.kit.livestream.app/primaryStream/signalPort=49100 \
  --/exts/omni.kit.livestream.app/primaryStream/streamPort=47998 \
  --/log/channels/omni.kit.livestream.streamsdk=info \
  --/log/channels/omni.kit.livestream.webrtc=info \
  --/log/channels/omni.kit.livestream.app=info
```

浏览器客户端必须使用相同地址：

```text
http://127.0.0.1:5173/?signalingServer=<UDP_REACHABLE_IP>&mediaServer=<UDP_REACHABLE_IP>&signalingPort=49100&mediaPort=47998
```

如果 5173 页面仍通过单独的 HTTP 代理访问，可以保留页面代理，但
`signalingServer/mediaServer` 必须显式指向真正的 TCP/UDP endpoint。

## 验收命令

服务启动完成后：

```bash
lsof -nP -iTCP:49100 -sTCP:LISTEN
lsof -nP -iUDP:47998
```

浏览器开始连接后，持续观察：

```bash
tail -F "/root/.nvidia-omniverse/logs/Kit/Isaac-Sim Streaming/6.0/"kit_*.log \
  | grep -Ei 'livestream|webrtc|streamsdk|candidate|client|error|failed'
```

最终验收不是“端口存在”，而是：

- 浏览器触发 `Stream Ready`；
- video element 开始接收帧；
- ETLI 出现 candidate pair、编码和收发统计；
- 47998/UDP 有实际媒体流量。
