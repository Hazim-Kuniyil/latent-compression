rsync -avz -P \
  -e "ssh -p 19714 -i ~/.ssh/id_ed25519" \
  root@74.2.96.22:~/latent-compression/outputs_stage2 \
  ./