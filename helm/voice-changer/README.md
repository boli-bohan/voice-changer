# Voice Changer Helm Chart

## Enabling a TURN server

The default deployment exposes the worker pods with a `NodePort` service so that
WebRTC peers can reach them directly. In production you can keep the worker
nodes private by adding a single TURN (Traversal Using Relays around NAT) relay
with a public IP address and configuring the chart to send all media through
that relay. The TURN server terminates the public traffic while the workers stay
on private cluster networking.

### 1. Enable the TURN components

Update your `values.yaml` (or pass `--set` flags) to turn on the Coturn
deployment bundled with the chart:

```yaml
turn:
  enabled: true
  service:
    type: LoadBalancer
    externalTrafficPolicy: Local
  relay:
    realm: voice-changer.example.com
    publicIP: 203.0.113.10  # Optional if your LoadBalancer assigns an address automatically
  credentials:
    static:
      username: rtc-relay
      password: super-secure-password
```

Key settings:

- `turn.service.type` should stay `LoadBalancer` (or `NodePort` if you plan to
  use an external static IP). Only the TURN deployment needs a public endpoint.
- `turn.relay.publicIP` is required when the LoadBalancer sits behind NAT (for
  example, on bare metal or certain cloud providers). Coturn advertises this
  address back to WebRTC clients.
- Replace the static credentials with values stored in a Secret in production by
  setting `turn.credentials.static.existingSecret` to the name of a pre-created
  secret that contains `username` and `password` keys.
- Optionally flip `turn.hostNetwork` to `true` if your provider cannot expose
  large UDP ranges through a Service. This binds Coturn directly to the node’s
  network interfaces so that the relay ports configured under
  `turn.ports.udp.min`/`max` are reachable.

### 2. Keep worker pods private

With TURN enabled you can change the worker Service type to `ClusterIP` so that
only in-cluster components reach them:

```yaml
worker:
  service:
    type: ClusterIP
```

The WebRTC clients (browsers or native apps) now authenticate with Coturn using
`username`/`password` and receive relay candidates that point back to the TURN
LoadBalancer. Coturn forwards the RTP/RTCP streams to the worker pods over the
private network, so no worker receives a public IP.

### 3. Distribute TURN configuration to clients

Expose the TURN credentials and URL (for example,
`turn:turn.voice-changer.example.com:3478`) through your signalling channel or
API so that browsers add the relay to their ICE server list:

```json
{
  "iceServers": [
    {
      "urls": ["turn:turn.voice-changer.example.com:3478"],
      "username": "rtc-relay",
      "credential": "super-secure-password"
    }
  ]
}
```

After these changes, only the TURN LoadBalancer requires a public IP; worker
pods remain on internal networking while still being reachable to end users
through the relay.
