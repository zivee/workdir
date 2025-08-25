#!/bin/sh
# 自动检测 WireGuard 隧道并刷新连接
# 运行环境：OpenWrt / Shell
# 定时任务： */5 * * * * /usr/bin/wg_keepalive.sh
# 查看 logread -f | grep wireguard

WG_IF="wg0"                         # WireGuard 接口
PEER_PUBKEY="xxxxxxxxxxxxxxxxxxxx"  # A 端公钥
ENDPOINT="vpn.example.com:51820"    # A 端域名:端口（必须是域名！）
CHECK_IP="10.0.10.1"                # 用于探测的 IP（A 端 wg0 地址或内网设备）
PING_COUNT=3                        # 连续 ping 几次
PING_FAIL_THRESHOLD=3               # 超过几次失败才刷新

# 检查连接
fail=0
for i in $(seq 1 $PING_COUNT); do
    if ! ping -c1 -W1 $CHECK_IP >/dev/null 2>&1; then
        fail=$((fail+1))
    fi
done

if [ $fail -ge $PING_FAIL_THRESHOLD ]; then
    logger -t wireguard "[$WG_IF] Peer unreachable ($CHECK_IP), refreshing endpoint..."
    # 刷新 endpoint（优雅方式）
    wg set $WG_IF peer $PEER_PUBKEY endpoint $ENDPOINT

    # 如果需要彻底重启接口，用下面两行替代
    # ifdown $WG_IF
    # ifup $WG_IF
else
    logger -t wireguard "[$WG_IF] Peer reachable ($CHECK_IP), OK."
fi
