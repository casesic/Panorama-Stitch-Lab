echo $SSH_CLIENT
cat > ~/.claude/settings.json << 'EOF'
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://172.28.208.226:15721",
    "ANTHROPIC_AUTH_TOKEN": "any-string"
  }
}
EOF
unset ANTHROPIC_API_KEY

netsh interface portproxy delete v4tov4 listenaddress=172.28.208.185 listenport=15721
netsh interface portproxy add v4tov4 listenaddress=172.28.208.226 connectaddress=127.0.0.1 connectport=15721 listenport=15721